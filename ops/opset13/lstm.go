package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinLSTMInputs = 3
	MaxLSTMInputs = 8
)

// LSTM represents the ONNX lstm operator.
type LSTM struct {
	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int
	inputForget     bool

	outputs []string
}

// newLSTM creates a new lstm operator.
func newLSTM() ops.Operator {
	return &LSTM{
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
		return nil, ops.ErrUnsupportedInput("sequence_lens", l)
	}

	X := inputs[0]

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
		nBiasMatrices := 8
		B = l.getZeroTensor(1, nBiasMatrices*l.hiddenSize)
	}

	Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc, err := l.getBiases(B)
	if err != nil {
		return nil, err
	}

	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Ht := inputs[5]
	if Ht == nil {
		Ht = l.getZeroTensor(1, batchSize, l.hiddenSize)
	}

	Ct := inputs[6]
	if Ct == nil {
		Ct = l.getZeroTensor(1, batchSize, l.hiddenSize)
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

	fActivation := ops.Activations[l.activations[0]]
	if fActivation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", l)
	}

	gActivation := ops.Activations[l.activations[1]]
	if gActivation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", l)
	}

	hActivation := ops.Activations[l.activations[2]]
	if hActivation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", l)
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

	// Reshape the outputs so it adds the num_directions as specified by onnx.
	// The output shape as specified by ONNX is:
	//   (sequence_length, num_directions, batch_size, hidden_size)
	// 'num_directions' is only '2' if the LSTMDirection is 'bidirectional'.
	// We do not support this, so for this implementation it should always be '1'.
	// Here, we reshape our output to include this 'num_directions' dimension.
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LSTM) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LSTM) GetMinInputs() int {
	return MinLSTMInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LSTM) GetMaxInputs() int {
	return MaxLSTMInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LSTM) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Int32},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LSTM) String() string {
	return "lstm operator"
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
	gemm := &Gemm{transA: false, transB: true, alpha: 1.0, beta: 1.0}

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
func (l *LSTM) getWeights(W tensor.Tensor) (Wi, Wo, Wf, Wh tensor.Tensor, err error) {
	nWeightMatrices := 4
	nWeightDimensions := 3

	weights, err := l.extractMatrices(W, nWeightMatrices, nWeightDimensions)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return weights[0], weights[1], weights[2], weights[3], nil
}

// getBiases splits tensor B into 8 bias matrices.
func (l *LSTM) getBiases(B tensor.Tensor) (Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc tensor.Tensor, err error) {
	nBiasMatrices := 8
	nBiasDimensions := 2

	b, err := l.extractMatrices(B, nBiasMatrices, nBiasDimensions)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, err
	}

	return b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], nil
}

// getPeepholes splits tensor P into 3 bias matrices.
func (l *LSTM) getPeepholes(P tensor.Tensor) (Pi, Po, Pf tensor.Tensor, err error) {
	nPeepholeMatrices := 3
	nPeepholeDimensions := 2

	p, err := l.extractMatrices(P, nPeepholeMatrices, nPeepholeDimensions)
	if err != nil {
		return nil, nil, nil, err
	}

	return p[0], p[1], p[2], nil
}

// extractMatrices extracts 4 tensors from tensor M.
// M contains all matrices concatenated on top of each other in the following order:
//
//	forward weights:   [Wi, Wo, Wf, Wc]
//	recurrent weights: [Ri, Ro, Rf, Rc]
//	biases:            [Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc]
//
// M is assumed to have a shape of (num_directions, nMatrices * hidden_size, ...) and we extract the
// by slicing over the 'nMatrices * hidden_size' dimension.
func (l *LSTM) extractMatrices(M tensor.Tensor, nMatrices, nDimensions int) ([]tensor.Tensor, error) {
	dirSlice := ops.NewSlicer(0)
	matrices := make([]tensor.Tensor, nMatrices)

	for i := 0; i < nMatrices; i++ {
		hiddenSlice := ops.NewSlicer(i*l.hiddenSize, (i+1)*l.hiddenSize)

		allSlices := make([]tensor.Slice, nDimensions)
		allSlices[0] = dirSlice
		allSlices[1] = hiddenSlice

		for i := 2; i < nDimensions; i++ {
			allSlices[i] = nil
		}

		m, err := M.Slice(allSlices...)
		if err != nil {
			return nil, err
		}

		matrices[i] = m
	}

	return matrices, nil
}

// getZeroTensor returns a tensor filled with zeros with the given shape.
func (l *LSTM) getZeroTensor(shape ...int) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(shape...),
		tensor.WithBacking(ops.Zeros(ops.NElements(shape...))),
	)
}
