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

// RNNDirection is the direction of the RNN. RNNs process sequences. We can process
// those forward (from first to last), in reverse (from last to first) or
// bidirectional (which is both forward and reverse added together).
type RNNDirection string

const (
	Forward       RNNDirection = "forward"
	Reverse       RNNDirection = "reverse"
	Bidirectional RNNDirection = "bidirectional"
)

// These activations are supported in the RNN calculation.
var RNNActivations = map[string]ops.Activation{
	"tanh":    ops.Tanh,
	"sigmoid": ops.Sigmoid,
	"relu":    ops.ReLU,
}

// RNN represents the ONNX rnn operator.
type RNN struct {
	activationAlpha []float32
	activationBeta  []float32
	activations     []string
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
			return ops.ErrUnsupportedAttribute(attr.GetName(), r)
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

	Ht := inputs[5]
	if Ht == nil {
		Ht = r.getInitialH(batchSize)
	}

	// Reshape the hidden tensor without the bidirectional dimension, as
	// we do not support bidirectional RNN yet. This is the dimension at
	// index 0.
	if err = Ht.Reshape(Ht.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	activation := RNNActivations[r.activations[0]]
	if activation == nil {
		return nil, ops.ErrUnsupportedAttribute("activations", r)
	}

	outputs := []tensor.Tensor{}

	// Loop over all timesteps of the input, applying the RNN calculation to every
	// timesteps while updating the hidden tensor.
	for t := 0; t < seqLength; t++ {
		Xt, err := X.Slice(ops.NewSlicer(t, t+1), nil, nil)
		if err != nil {
			return nil, err
		}

		Ht, err = r.layerCalculation(Xt, Ht, Wi, Ri, Wbi, Rbi, RNNActivations[r.activations[0]])
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

	// Reshape the outputs so it adds the num_directions as specified by onnx.
	// The output shape as specified by ONNX is:
	//   (sequence_length, num_directions, batch_size, hidden_size)
	// 'num_directions' is only '2' if the RNNDirection is 'bidirectional'.
	// We do not support this, so for this implementation it should always be '1'.
	// Here, we reshape our output to include this 'num_directions' dimension.
	if err = Y.Reshape(seqLength, 1, batchSize, r.hiddenSize); err != nil {
		return nil, err
	}

	if err = Yh.Reshape(1, batchSize, r.hiddenSize); err != nil {
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

// getBiases splits an input bias tensor B into its subparts. The B input for the
// RNN operator consists of two biases, Wbi and Rbi. These biases are concatenated
// in the second dimension, where B has shape (num_directions, 2 * hiddenSize).
// This function slices the B tensor to return 2 bias tensors. We disregard the
// num_directions axis as we do not support the bidirectional direction.
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
// By default all values are 0. Note that we do not support the bidirectional
// option so the first dim always has size 1.
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
// As we do not support the birectional option, the num_directions dim size is
// always 1.
func (r *RNN) getInitialH(batchSize int) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(1, batchSize, r.hiddenSize),
		tensor.WithBacking(ops.Zeros(batchSize*r.hiddenSize)),
	)
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
