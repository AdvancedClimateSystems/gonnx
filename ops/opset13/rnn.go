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
	return &RNN{}
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
	X := inputs[0]
	W := inputs[1]
	R := inputs[2]
	B := inputs[3]

	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", r)
	}

	initialH := inputs[5]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wz, Wr, Wh, err := r.getForwardWeights(W)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *RNN) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
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
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *RNN) String() string {
	return "rnn operator"
}

// getForwardWeights returns the weights for the gate.
func (r *RNN) getForwardWeights(W tensor.Tensor) (Wz, Wr, Wh tensor.Tensor, err error) {
	n, err := r.extractWeights(W)
	if err != nil {
		return nil, nil, nil, err
	}

	return n[0], n[1], n[2], nil
}

// extractWeights extracts 1 or 2 weight tensors from node W.
// W contains all 2 weight tensors concatenated on top of each other in the following order:
//
//	forward weights:   [Wi, Wbi]
//	recurrent weights: [Ri, Rbi]
//
// W will have a shape of (num_directions, 2 * hidden_size, ...) and we extract the
// by slicing over the '2 * hidden_size' dimension.
func (r *RNN) extractWeights(W tensor.Tensor) ([]tensor.Tensor, error) {
	nWeightMatrices := 1
	if r.direction == Bidirectional {
		nWeightMatrices = 2
	}

	dirSlice := ops.NewSlicer(0)
	weights := make([]tensor.Tensor, nWeightMatrices)

	for i := 0; i < nWeightMatrices; i++ {
		slice := ops.NewSlicer(i*r.hiddenSize, (i+1)*r.hiddenSize)

		w, err := W.Slice(dirSlice, slice, nil)
		if err != nil {
			return nil, err
		}

		weights[i] = w
	}

	return weights, nil
}
