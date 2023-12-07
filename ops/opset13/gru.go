package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinGRUInputs = 3
	MaxGRUInputs = 6
)

// GRU represents the ONNX gru operator. It only supports a simple forward gru
// operation with default activations.
type GRU struct {
	activationAlpha   []float32
	activationBeta    []float32
	activations       []string
	direction         ops.SequenceProcessDirection
	hiddenSize        int
	linearBeforeReset bool
}

// newGRU creates a new gru operator.
func newGRU() ops.Operator {
	return &GRU{
		activations:       []string{"sigmoid", "tanh"},
		direction:         ops.Forward,
		linearBeforeReset: false,
	}
}

// Init initializes the gru operator. Currently, our GRU operator does not support all
// attributes as specified by the ONNX operator. The basic functionality is working and
// the other attributes can be added later on.
func (g *GRU) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			g.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			g.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			g.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), g)
		case ops.DirectionAttr:
			g.direction = ops.SequenceProcessDirection(attr.GetS())
			if g.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), g)
			}
		case ops.HiddenSizeAttr:
			g.hiddenSize = int(attr.GetI())
		case "linear_before_reset":
			g.linearBeforeReset = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the gru operator.
func (g *GRU) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	X := inputs[0]
	W := inputs[1]
	R := inputs[2]
	B := inputs[3]

	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", g)
	}

	initialH := inputs[5]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wz, Wr, Wh, err := g.getForwardWeights(W)
	if err != nil {
		return nil, err
	}

	Rz, Rr, Rh, err := g.getRecurrentWeights(R)
	if err != nil {
		return nil, err
	}

	if B == nil {
		B = g.initialB()
	}

	Wbz, Wbr, Wbh, Rbz, Rbr, Rbh, err := g.getBiases(B)
	if err != nil {
		return nil, err
	}

	var prevH tensor.Tensor
	if initialH == nil {
		prevH = g.initialH(batchSize)
	} else {
		var ok bool
		prevH, ok = initialH.Clone().(tensor.Tensor)
		if !ok {
			return nil, ops.ErrTypeAssert("tensor.Tensor", initialH.Clone())
		}
	}

	// Extract the shape of the hidden dimensions without the bidirectional dimension, as
	// we do not support bidirectional GRU yet.
	shapeWithoutBidir := prevH.Shape().Clone()[1:]

	err = prevH.Reshape(shapeWithoutBidir...)
	if err != nil {
		return nil, err
	}

	fActivation := ops.Activations[g.activations[0]]
	if fActivation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", g)
	}

	gActivation := ops.Activations[g.activations[1]]
	if gActivation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", g)
	}

	outputs := []tensor.Tensor{}

	for i := 0; i < seqLength; i++ {
		Xt, err := g.extractXt(X, i)
		if err != nil {
			return nil, err
		}

		zt, err := g.gateCalculation(Xt, prevH, Wz, Rz, Wbz, Rbz, fActivation)
		if err != nil {
			return nil, err
		}

		rt, err := g.gateCalculation(Xt, prevH, Wr, Rr, Wbr, Rbr, fActivation)
		if err != nil {
			return nil, err
		}

		ht, err := g.htCalculation(Xt, prevH, rt, Wh, Rh, Wbh, Rbh, gActivation)
		if err != nil {
			return nil, err
		}

		prevH, err = g.hiddenCalculation(zt, ht, prevH)
		if err != nil {
			return nil, err
		}

		outputs = append(outputs, prevH)
	}

	var Y tensor.Tensor
	if len(outputs) > 1 {
		Y, err = tensor.Concat(0, outputs[0], outputs[1:]...)
		if err != nil {
			return nil, err
		}
	} else {
		Y = outputs[0]
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Y.Reshape([]int{seqLength, 1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	Yh, ok := prevH.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", prevH.Clone())
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Yh.Reshape([]int{1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *GRU) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *GRU) GetMinInputs() int {
	return MinGRUInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *GRU) GetMaxInputs() int {
	return MaxGRUInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *GRU) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Int32},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *GRU) String() string {
	return "gru operator"
}

// extractXt extracts the value of x for timestep t.
func (g *GRU) extractXt(X tensor.Tensor, t int) (tensor.Tensor, error) {
	return X.Slice(ops.NewSlicer(t, t+1), nil, nil)
}

func (g *GRU) gateCalculation(
	Xt, H, W, R, Wb, Rb tensor.Tensor, activation ops.Activation,
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

	gate, err := tensor.Add(inputCalc[0], hiddenCalc[0])
	if err != nil {
		return nil, err
	}

	return activation(gate)
}

func (g *GRU) htCalculation(
	Xt, prevH, rt, W, R, Wb, Rb tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	if !g.linearBeforeReset {
		temp1, err := tensor.Mul(rt, prevH)
		if err != nil {
			return nil, err
		}

		return g.gateCalculation(Xt, temp1, W, R, Wb, Rb, activation)
	}

	gemm := &Gemm{transA: false, transB: true, alpha: 1.0, beta: 1.0}

	inputCalc, err := gemm.Apply([]tensor.Tensor{Xt, W, Wb})
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := gemm.Apply([]tensor.Tensor{prevH, R, Rb})
	if err != nil {
		return nil, err
	}

	temp1, err := tensor.Mul(hiddenCalc[0], rt)
	if err != nil {
		return nil, err
	}

	temp2, err := tensor.Add(temp1, inputCalc[0])
	if err != nil {
		return nil, err
	}

	return activation(temp2)
}

func (g *GRU) hiddenCalculation(zt, ht, prevH tensor.Tensor) (tensor.Tensor, error) {
	temp1, err := tensor.Sub(onesTensor(zt), zt)
	if err != nil {
		return nil, err
	}

	temp2, err := tensor.Mul(temp1, ht)
	if err != nil {
		return nil, err
	}

	temp3, err := tensor.Mul(zt, prevH)
	if err != nil {
		return nil, err
	}

	return tensor.Add(temp2, temp3)
}

// getForwardWeights returns the weights for the gate.
func (g *GRU) getForwardWeights(W tensor.Tensor) (Wz, Wr, Wh tensor.Tensor, err error) {
	n, err := g.extractWeights(W)
	if err != nil {
		return nil, nil, nil, err
	}

	return n[0], n[1], n[2], nil
}

// getRecurrentWeights returns recurrent weights.
func (g *GRU) getRecurrentWeights(R tensor.Tensor) (Rz, Rr, Rh tensor.Tensor, err error) {
	recurrentWeights, err := g.extractWeights(R)
	if err != nil {
		return nil, nil, nil, err
	}

	return recurrentWeights[0], recurrentWeights[1], recurrentWeights[2], nil
}

// getBiases returns the biases from the Bias node as specified by the ONNX standard.
func (g *GRU) getBiases(B tensor.Tensor) (Wbz, Wbr, Wbh, Rbz, Rbr, Rbh tensor.Tensor, err error) {
	biases, err := g.extractBiases(B)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return biases[0], biases[1], biases[2], biases[3], biases[4], biases[5], nil
}

// extractWeights extracts 3 weight tensors from node W.
// W contains all 3 weight tensors concatenated on top of each other in the following order:
//
//	forward weights:   [Wz, Wr, Wh]
//	recurrent weights: [Rz, Rr, Rh]
//
// W will have a shape of (num_directions, 3 * hidden_size, ...) and we extract the
// by slicing over the '3 * hidden_size' dimension.
func (g *GRU) extractWeights(W tensor.Tensor) ([]tensor.Tensor, error) {
	const nWeightMatrices = 3

	dirSlice := ops.NewSlicer(0)
	weights := make([]tensor.Tensor, nWeightMatrices)

	for i := 0; i < nWeightMatrices; i++ {
		slice := ops.NewSlicer(i*g.hiddenSize, (i+1)*g.hiddenSize)

		w, err := W.Slice(dirSlice, slice, nil)
		if err != nil {
			return nil, err
		}

		weights[i] = w
	}

	return weights, nil
}

// extractBiases extracts the 6 bias tensors from tensor B.
// B contains all 6 bias tensors concatenated on top of each other in the following order:
//
//	[Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
//
// B has a shape of (num_directions, 6 * hidden_size) and every individual bias tensor should have
// shape (hidden_size). We extract the biases by slicing over the '6 * hidden_size' dimension.
func (g *GRU) extractBiases(B tensor.Tensor) ([]tensor.Tensor, error) {
	const nWeightMatrices = 6

	dirSlice := ops.NewSlicer(0)
	biases := make([]tensor.Tensor, nWeightMatrices)

	for i := 0; i < nWeightMatrices; i++ {
		slice := ops.NewSlicer(i*g.hiddenSize, (i+1)*g.hiddenSize)

		w, err := B.Slice(dirSlice, slice)
		if err != nil {
			return nil, err
		}

		biases[i] = w
	}

	return biases, nil
}

// initialB returns the initialB tensor. If the biases are not specified by the inputs
// of the gru operator this tensor can be used as the biases tensor. By default biases
// are all 0.
func (g *GRU) initialB() tensor.Tensor {
	const nWeightMatrices = 6

	return tensor.New(
		tensor.WithShape(1, nWeightMatrices*g.hiddenSize),
		tensor.WithBacking(ops.Zeros(nWeightMatrices*g.hiddenSize)),
	)
}

// initialH can be used for initialH when it is not specified by the inputs of the operator.
// if it is not specified by the inputs assumed to be 0. It has shape
// (num_directions, batch_size, hidden_size).
func (g *GRU) initialH(batchSize int) tensor.Tensor {
	hiddenFloats := ops.Zeros(batchSize * g.hiddenSize)

	return tensor.New(
		tensor.WithShape(1, batchSize, g.hiddenSize),
		tensor.WithBacking(hiddenFloats),
	)
}

// onesTensor returns a new tensor with the same shape as the given tensor intialized with all ones.
func onesTensor(t tensor.Tensor) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(t.Shape()...),
		tensor.WithBacking(ops.Ones(ops.NElements(t.Shape()...))),
	)
}
