package lstm

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/gemm"
	"gorgonia.org/tensor"
)

const (
	MinLSTMInputs = 3
	MaxLSTMInputs = 8
)

var lstmTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

// LSTM represents the ONNX lstm operator.
type LSTM struct {
	ops.BaseOperator

	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int
	inputForget     bool

	outputs []string
}

// newLSTM creates a new lstm operator.
func newLSTM(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &LSTM{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinLSTMInputs,
			MaxLSTMInputs,
			typeConstraints,
			"lstm",
		),
		activations: []string{"sigmoid", "tanh", "tanh"},
		direction:   ops.Forward,
		inputForget: false,
		outputs:     []string{"Y", "Y_h", "Y_c"},
	}
}

// Init initializes the lstm operator.
func (l *LSTM) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			l.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			l.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			l.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), l)
		case ops.DirectionAttr:
			l.direction = ops.SequenceProcessDirection(attr.GetS())
			if l.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), l)
			}
		case ops.HiddenSizeAttr:
			l.hiddenSize = int(attr.GetI())
		case "input_forget":
			l.inputForget = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	l.outputs = n.GetOutput()

	return nil
}

// Apply applies the lstm operator.
func (l *LSTM) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence_lens", l.BaseOperator)
	}

	X := inputs[0]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wi, Wo, Wf, Wc, err := l.getWeights(inputs[1])
	if err != nil {
		return nil, err
	}

	Ri, Ro, Rf, Rc, err := l.getWeights(inputs[2])
	if err != nil {
		return nil, err
	}

	B := inputs[3]
	if B == nil {
		// 8 is the number of bias matrices required by ONNX definition.
		nBiasMatrices := 8
		B = ops.ZeroTensor(1, nBiasMatrices*l.hiddenSize)
	}

	Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc, err := l.getBiases(B)
	if err != nil {
		return nil, err
	}

	Ht := inputs[5]
	if Ht == nil {
		Ht = ops.ZeroTensor(1, batchSize, l.hiddenSize)
	}

	Ct := inputs[6]
	if Ct == nil {
		Ct = ops.ZeroTensor(1, batchSize, l.hiddenSize)
	}

	var Pi, Po, Pf tensor.Tensor

	P := inputs[7]
	if P != nil {
		Pi, Po, Pf, err = l.getPeepholes(P)
		if err != nil {
			return nil, err
		}
	}

	// Reshape the hidden and cell tensor without the bidirectional dimension, as
	// we do not support bidirectional yet. This is the dimension at
	// index 0.
	if err = Ht.Reshape(Ht.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	if err = Ct.Reshape(Ct.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	fActivation, err := ops.GetActivation(l.activations[0])
	if err != nil {
		return nil, err
	}

	gActivation, err := ops.GetActivation(l.activations[1])
	if gActivation == nil {
		return nil, err
	}

	hActivation, err := ops.GetActivation(l.activations[2])
	if err != nil {
		return nil, err
	}

	outputs := []tensor.Tensor{}

	// Loop over all timesteps of the input, applying the LSTM calculation to every
	// timesteps while updating the hidden tensor.
	for t := 0; t < seqLength; t++ {
		Xt, err := X.Slice(ops.NewSlicer(t, t+1), nil, nil)
		if err != nil {
			return nil, err
		}

		it, err := l.gateCalculation(Xt, Wi, Wbi, Ht, Ri, Rbi, Pi, Ct, fActivation)
		if err != nil {
			return nil, err
		}

		ft, err := l.gateCalculation(Xt, Wf, Wbf, Ht, Rf, Rbf, Pf, Ct, fActivation)
		if err != nil {
			return nil, err
		}

		ct, err := l.gateCalculation(Xt, Wc, Wbc, Ht, Rc, Rbc, nil, nil, gActivation)
		if err != nil {
			return nil, err
		}

		Ct, err = l.cellCalculation(ft, it, ct, Ct)
		if err != nil {
			return nil, err
		}

		ot, err := l.gateCalculation(Xt, Wo, Wbo, Ht, Ro, Rbo, Po, Ct, fActivation)
		if err != nil {
			return nil, err
		}

		Ht, err = l.hiddenCalculation(ot, Ct, hActivation)
		if err != nil {
			return nil, err
		}

		outputs = append(outputs, Ht)
	}

	Y := outputs[0]
	if len(outputs) > 1 {
		Y, err = tensor.Concat(0, Y, outputs[1:]...)
		if err != nil {
			return nil, err
		}
	}

	Yh, ok := Ht.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", Ht.Clone())
	}

	Yc, ok := Ct.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", Ct.Clone())
	}

	// Reshape the hidden tensor without the bidirectional dimension, as
	// we do not support bidirectional RNN yet. This is the dimension at
	// index 0.
	if err = Y.Reshape(seqLength, 1, batchSize, l.hiddenSize); err != nil {
		return nil, err
	}

	if err = Yh.Reshape(1, batchSize, l.hiddenSize); err != nil {
		return nil, err
	}

	if err = Yc.Reshape(1, batchSize, l.hiddenSize); err != nil {
		return nil, err
	}

	outputMap := map[string]tensor.Tensor{
		"Y": Y, "Y_h": Yh, "Y_c": Yc,
	}

	result := []tensor.Tensor{}
	for _, outputName := range l.outputs {
		result = append(result, outputMap[outputName])
	}

	return result, nil
}

// gateCalculation performs a standard gate calculation for an LSTM gate defined as:
//
//	o = f(Xt*(W^T) + Wb + H*(R^T) + Rb + P (.) C)
//
// Where:
//   - 'f()' is an activation function
//   - 'Xt' is the input tensor
//   - 'W' is the input weight
//   - 'Wb' is the input bias
//   - 'H' is the hidden tensor
//   - 'R' is the hidden weight tensor
//   - 'Rb' is the hidden bias
//   - 'P' are peephole weights (optional, can be nil)
//   - 'C' is the cell state
//   - '(.)' is element-wise multiplication
//
// 'o' is the result tensor that is returned.
// This calculation can be used for the forget gate, input gate, cell gate
// and output gate calculations.
func (l *LSTM) gateCalculation(
	Xt, W, Wb, H, R, Rb, P, C tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	gemm := gemm.GetVersions()[13]()

	err := gemm.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "alpha", F: 1.0},
				{Name: "beta", F: 1.0},
				{Name: "transA", I: 0},
				{Name: "transB", I: 1},
			},
		},
	)
	if err != nil {
		return nil, err
	}

	inputCalc, err := gemm.Apply([]tensor.Tensor{Xt, W, Wb})
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := gemm.Apply([]tensor.Tensor{H, R, Rb})
	if err != nil {
		return nil, err
	}

	output, err := tensor.Add(inputCalc[0], hiddenCalc[0])
	if err != nil {
		return nil, err
	}

	if P != nil {
		C, broadcastedP, err := ops.UnidirectionalBroadcast(C, P)
		if err != nil {
			return nil, err
		}

		peepholeActivation, err := tensor.Mul(broadcastedP, C)
		if err != nil {
			return nil, err
		}

		output, err = tensor.Add(output, peepholeActivation)
		if err != nil {
			return nil, err
		}
	}

	return activation(output)
}

// cellCalculation performs the calculation of the LSTM cell update defined by:
//
//	Ct = ft (.) Ct-1 + it (.) ct
//
// Where 'ft' is the forget gate activation at time t, (.) denotes element-wise
// multiplication, 'Ct-1' denotes the cell state at time t-1, 'it' denotes the input
// gate activation at time t and 'ct' denotes the cell state activation at time t (which)
// is not the same as Ct or Ct-1).
func (l *LSTM) cellCalculation(ft, it, ct, Ct tensor.Tensor) (tensor.Tensor, error) {
	cellForget, err := tensor.Mul(ft, Ct)
	if err != nil {
		return nil, err
	}

	cellInput, err := tensor.Mul(it, ct)
	if err != nil {
		return nil, err
	}

	return tensor.Add(cellForget, cellInput)
}

// hiddenCalculation performs the calculation of the new LSTM hidden state defined by:
//
//	Ht = ot (.) h(Ct)
//
// Where Ht is the new hidden state at time t, 'ot' is the output at time t, (.) denotes
// element-wise multiplication, 'h()' denotes an activation function and 'Ct' denotes the
// cell state at time t.
func (l *LSTM) hiddenCalculation(ot, Ct tensor.Tensor, activation ops.Activation) (tensor.Tensor, error) {
	cellActivated, err := activation(Ct)
	if err != nil {
		return nil, err
	}

	return tensor.Mul(ot, cellActivated)
}

// getWeights splits tensor W into 4 weight matrices.
// The W tensor, by GONNX definition, has 3 dimensions with 4 weight
// tensors in it (8 if bidirectional, but that is not supported).
func (l *LSTM) getWeights(W tensor.Tensor) (Wi, Wo, Wf, Wh tensor.Tensor, err error) {
	nWeightMatrices := 4
	nWeightDimensions := 3

	weights, err := ops.ExtractMatrices(W, nWeightMatrices, nWeightDimensions, l.hiddenSize)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return weights[0], weights[1], weights[2], weights[3], nil
}

// getBiases splits tensor B into 8 bias matrices.
// The B tensor, by GONNX definition, has 2 dimensions with 8 bias
// tensors in it (16 if bidirectional, but that is not supported).
func (l *LSTM) getBiases(B tensor.Tensor) (Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc tensor.Tensor, err error) {
	nBiasMatrices := 8
	nBiasDimensions := 2

	b, err := ops.ExtractMatrices(B, nBiasMatrices, nBiasDimensions, l.hiddenSize)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}

	return b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], nil
}

// getPeepholes splits tensor P into 3 bias matrices.
// The P tensor, by GONNX definition, has 2 dimensions with 3 peephole
// tensors in it (6 if bidirectional, but that is not supported).
func (l *LSTM) getPeepholes(P tensor.Tensor) (Pi, Po, Pf tensor.Tensor, err error) {
	nPeepholeMatrices := 3
	nPeepholeDimensions := 2

	p, err := ops.ExtractMatrices(P, nPeepholeMatrices, nPeepholeDimensions, l.hiddenSize)
	if err != nil {
		return nil, nil, nil, err
	}

	return p[0], p[1], p[2], nil
}
