package rnn

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/gemm"
	"gorgonia.org/tensor"
)

var rnnTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
}

const (
	MinRNNInputs = 3
	MaxRNNInputs = 6
)

// RNN represents the ONNX rnn operator.
type RNN struct {
	ops.BaseOperator

	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int
	layout          int
}

// newRNN creates a new rnn operator.
func newRNN(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &RNN{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinRNNInputs,
			MaxRNNInputs,
			typeConstraints,
			"rnn",
		),
		activations: []string{"tanh"},
		direction:   ops.Forward,
		layout:      0,
	}
}

// Init initializes the rnn operator.
func (r *RNN) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			r.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			r.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			r.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), r)
		case ops.DirectionAttr:
			r.direction = ops.SequenceProcessDirection(attr.GetS())
			if r.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), r)
			}
		case ops.HiddenSizeAttr:
			r.hiddenSize = int(attr.GetI())
		case ops.LayoutAttr:
			// 'layout' is supported since version 14
			if r.Version() < 14 {
				return ops.ErrInvalidAttribute(attr.GetName(), r)
			}

			r.layout = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the rnn operator.
func (r *RNN) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", r.BaseOperator)
	}

	X := inputs[0]

	var seqLength int

	var batchSize int

	// The 'layout' parameter handles whether or not the batch dimension comes
	// first in the tensor. If this is the case, we reshape it here in
	// in the beginning of the operation, and reverse it at the end of the operation.
	if r.layout == 1 {
		seqLength = X.Shape()[1]
		batchSize = X.Shape()[0]
		inputSize := X.Shape()[2]

		err := X.Reshape(seqLength, batchSize, inputSize)
		if err != nil {
			return nil, err
		}
	} else {
		seqLength = X.Shape()[0]
		batchSize = X.Shape()[1]
	}

	Wi, err := r.getWeights(inputs[1])
	if err != nil {
		return nil, err
	}

	Ri, err := r.getWeights(inputs[2])
	if err != nil {
		return nil, err
	}

	B := inputs[3]
	if B == nil {
		// 2 is the number of bias matrices required by ONNX definition.
		nBiasMatrices := 2
		B = ops.ZeroTensor(1, nBiasMatrices*r.hiddenSize)
	}

	Wbi, Rbi, err := r.getBiases(B)
	if err != nil {
		return nil, err
	}

	Ht := inputs[5]
	if Ht == nil {
		if r.layout == 1 {
			Ht = ops.ZeroTensor(batchSize, 1, r.hiddenSize)
		} else {
			Ht = ops.ZeroTensor(1, batchSize, r.hiddenSize)
		}
	}

	// If layout is 1, this means batch size comes as first dimension, and
	// we reshape it here to the default layout.
	if r.layout == 1 {
		numDirections := Ht.Shape()[1]
		if err = Ht.Reshape(numDirections, batchSize, r.hiddenSize); err != nil {
			return nil, err
		}
	}

	// Reshape the hidden tensor without the bidirectional dimension, as
	// we do not support bidirectional RNN yet. This is the dimension at
	// index 0.
	if err = Ht.Reshape(Ht.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	activation, err := ops.GetActivation(r.activations[0])
	if err != nil {
		return nil, err
	}

	outputs := []tensor.Tensor{}

	// Loop over all timesteps of the input, applying the RNN calculation to every
	// timesteps while updating the hidden tensor.
	for t := 0; t < seqLength; t++ {
		Xt, err := X.Slice(ops.NewSlicer(t, t+1), nil, nil)
		if err != nil {
			return nil, err
		}

		Ht, err = r.layerCalculation(Xt, Ht, Wi, Ri, Wbi, Rbi, activation)
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

	// Reshape the output according to the specified layout and re-add the
	// num_directions dimension.
	if r.layout == 1 {
		if err = Y.Reshape(batchSize, seqLength, 1, r.hiddenSize); err != nil {
			return nil, err
		}

		if err = Yh.Reshape(batchSize, 1, r.hiddenSize); err != nil {
			return nil, err
		}
	} else {
		if err = Y.Reshape(seqLength, 1, batchSize, r.hiddenSize); err != nil {
			return nil, err
		}

		if err = Yh.Reshape(1, batchSize, r.hiddenSize); err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// layerCalculation performs the actual RNN calculation. By ONNX definition
// this is:
//
//	Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
//
// We achieve this by two Gemm operations, adding them together and finally
// putting them through an activation function.
func (r *RNN) layerCalculation(
	Xt, H, Wi, Ri, Wbi, Rbi tensor.Tensor, activation ops.Activation,
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

	inputCalc, err := gemm.Apply([]tensor.Tensor{Xt, Wi, Wbi})
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := gemm.Apply([]tensor.Tensor{H, Ri, Rbi})
	if err != nil {
		return nil, err
	}

	result, err := tensor.Add(inputCalc[0], hiddenCalc[0])
	if err != nil {
		return nil, err
	}

	return activation(result)
}

// getWeights returns the weights from a concatenated weight tensor. The result is
// a single weight matrix. W has shape (num_directions, hidden_size, ...).
// The W tensor, by GONNX definition, has 3 dimensions with 1 weight
// tensor in it (2 if bidirectional, but that is not supported).
func (r *RNN) getWeights(W tensor.Tensor) (tensor.Tensor, error) {
	nWeightMatrices := 1
	nWeightDimensions := 3

	weights, err := ops.ExtractMatrices(W, nWeightMatrices, nWeightDimensions, r.hiddenSize)
	if err != nil {
		return nil, err
	}

	return weights[0], nil
}

// getBiases splits tensor B into 2 bias matrices.
// The B tensor, by GONNX definition, has 2 dimensions with 2 bias
// tensors in it (4 if bidirectional, but that is not supported).
func (r *RNN) getBiases(B tensor.Tensor) (Wbi, Rbi tensor.Tensor, err error) {
	nBiasMatrices := 2
	nBiasDimensions := 2

	b, err := ops.ExtractMatrices(B, nBiasMatrices, nBiasDimensions, r.hiddenSize)
	if err != nil {
		return nil, nil, err
	}

	return b[0], b[1], nil
}
