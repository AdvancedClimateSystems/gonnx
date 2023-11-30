package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinRNNInputs = 3
	MaxRNNInputs = 6
)

type RNNDirection string

const (
	Forward       RNNDirection = "forward"
	Reverse       RNNDirection = "reverse"
	Bidirectional RNNDirection = "bidirectional"
)

var RNNActivations = map[string]ops.Activation{
	"Tanh":    ops.Tanh,
	"Sigmoid": ops.Tanh,
	"ReLU":    ops.ReLU,
}

// RNN represents the ONNX rnn operator.
type RNN struct {
	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	clip            float32
	direction       RNNDirection
	hiddenSize      int
}

// newRNN creates a new rnn operator.
func newRNN() ops.Operator {
	return &RNN{
		activations: []string{"tanh"},
		direction:   Forward,
	}
}

// Init initializes the rnn operator.
func (r *RNN) Init(attributes []*onnx.AttributeProto) error {
	for _, attr := range attributes {
		switch attr.GetName() {
		case "activation_alpha":
			r.activationAlpha = attr.GetFloats()
		case "activation_beta":
			r.activationBeta = attr.GetFloats()
		case "activations":
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			r.activations = activations
		case "clip":
			r.clip = attr.GetF()
		case "direction":
			r.direction = RNNDirection(attr.GetS())
			if r.direction != Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), r)
			}
		case "hidden_size":
			r.hiddenSize = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the rnn operator.
func (r *RNN) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", r)
	}

	X := inputs[0]

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
		B = r.getDefaultB()
	}

	Wbi, Rbi, err := r.getBiases(B)
	if err != nil {
		return nil, err
	}

	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	prevH := inputs[5]
	if prevH == nil {
		prevH = r.getInitialH(batchSize)
	}

	// Extract the shape of the hidden dimensions without the bidirectional dimension, as
	// we do not support bidirectional RNN yet.
	if err = prevH.Reshape(prevH.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	outputs := []tensor.Tensor{}

	for t := 0; t < seqLength; t++ {
		Xt, err := X.Slice(ops.NewSlicer(t, t+1), nil, nil)
		if err != nil {
			return nil, err
		}

		prevH, err = r.layerCalculation(Xt, prevH, Wi, Ri, Wbi, Rbi, ops.Tanh)
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
	err = Y.Reshape(seqLength, 1, batchSize, r.hiddenSize)
	if err != nil {
		return nil, err
	}

	Yh, ok := prevH.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", prevH.Clone())
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Yh.Reshape(1, batchSize, r.hiddenSize)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *RNN) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *RNN) GetMinInputs() int {
	return MinRNNInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *RNN) GetMaxInputs() int {
	return MaxRNNInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *RNN) GetInputTypeConstraints() [][]tensor.Dtype {
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
func (r *RNN) String() string {
	return "rnn operator"
}

// getWeights returns the weights from a concatenated weight tensor. The result is
// a single weight matrix. W has shape (num_directions, hidden_size, ...).
// We do not support bidirectional layers, so we can simply index the first element
// of W to get the weights for either the input or the recurrence.
func (r *RNN) getWeights(X tensor.Tensor) (tensor.Tensor, error) {
	weights, err := X.Slice(ops.NewSlicer(0), nil, nil)
	if err != nil {
		return nil, err
	}

	return weights, nil
}

func (r *RNN) getBiases(B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	Wbi, err := B.Slice(ops.NewSlicer(0), ops.NewSlicer(0, r.hiddenSize))
	if err != nil {
		return nil, nil, err
	}

	nBiasMatrices := 2

	Rbi, err := B.Slice(ops.NewSlicer(0), ops.NewSlicer(r.hiddenSize, nBiasMatrices*r.hiddenSize))
	if err != nil {
		return nil, nil, err
	}

	return Wbi, Rbi, nil
}

// getDefaultB returns the default bias tensor if no bias tensor is provided.
// The bias tensor for RNN consists of two concatenated bias tensors, one for
// the input calculation and one for the hidden calculation. It has shape:
//
//	(num_directions, 2*hiddenSize).
//
// By default all values are 0.
func (r *RNN) getDefaultB() tensor.Tensor {
	nBiasMatrices := 2

	return tensor.New(
		tensor.WithShape(1, nBiasMatrices*r.hiddenSize),
		tensor.WithBacking(ops.Zeros(nBiasMatrices*r.hiddenSize)),
	)
}

// getInitialH can be used to construct an initial hidden tensor when it is not
// specified by the inputs of the operator. In this case it is assumed to be 0.
// It has shape (num_directions, batch_size, hidden_size).
func (r *RNN) getInitialH(batchSize int) tensor.Tensor {
	hiddenFloats := ops.Zeros(batchSize * r.hiddenSize)

	return tensor.New(
		tensor.WithShape(1, batchSize, r.hiddenSize),
		tensor.WithBacking(hiddenFloats),
	)
}

func (r *RNN) layerCalculation(
	Xt, H, Wi, Ri, Wbi, Rbi tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	gemm := &Gemm{transA: false, transB: true, alpha: 1.0, beta: 1.0}

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
