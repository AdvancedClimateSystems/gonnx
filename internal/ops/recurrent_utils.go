package ops

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/tensor"
)

// SequenceProcessDirection is the direction in which a sequential input is processed.
// We can process sequential inputs forward (from first to last), in reverse (from
// last to first) or bidirectional (which is both forward and reverse added together).
type SequenceProcessDirection string

const (
	Forward       SequenceProcessDirection = "forward"
	Reverse       SequenceProcessDirection = "reverse"
	Bidirectional SequenceProcessDirection = "bidirectional"
)

// These constants define attributes that are applicable to GRU, LSTM and RNN operators.
const (
	ActivationAlphaAttr = "activation_alpha"
	ActivationBetaAttr  = "activation_beta"
	ActivationsAttr     = "activations"
	ClipAttr            = "clip"
	DirectionAttr       = "direction"
	HiddenSizeAttr      = "hidden_size"
)

// RecurrentDims are the dimensions of the input of a recurrent operator.
type RecurrentDims struct {
	SeqLength int
	BatchSize int
	InputSize int
}

// ValidateRecurrentInputs checks the shapes and data types of the inputs of a recurrent
// operator (RNN, GRU or LSTM) with nGates gates per direction, and returns the dimensions
// of input X. The optional inputs B and initialH may be nil.
func ValidateRecurrentInputs(X, W, R, B, initialH *tensor.Tensor, nGates, hiddenSize int) (RecurrentDims, error) {
	if X.Rank() != 3 {
		return RecurrentDims{}, fmt.Errorf("%w: X must have shape [seq_length, batch_size, input_size], got %v", ErrInvalidShape, X.Shape())
	}

	dims := RecurrentDims{SeqLength: X.Shape()[0], BatchSize: X.Shape()[1], InputSize: X.Shape()[2]}

	expected := []struct {
		name  string
		t     *tensor.Tensor
		shape tensor.Shape
	}{
		{"W", W, tensor.Shape{1, nGates * hiddenSize, dims.InputSize}},
		{"R", R, tensor.Shape{1, nGates * hiddenSize, hiddenSize}},
		{"B", B, tensor.Shape{1, 2 * nGates * hiddenSize}},
		{"initial_h", initialH, tensor.Shape{1, dims.BatchSize, hiddenSize}},
	}

	for _, input := range expected {
		if input.t == nil {
			continue
		}

		if !input.t.Shape().Equal(input.shape) {
			return RecurrentDims{}, fmt.Errorf("%w: %s must have shape %v, got %v", ErrInvalidShape, input.name, input.shape, input.t.Shape())
		}

		if input.t.DataType() != X.DataType() {
			return RecurrentDims{}, ErrDataTypeMismatch(X.DataType(), input.t.DataType())
		}
	}

	return dims, nil
}

// RecurrentBiases returns the input (Wb) and recurrent (Rb) biases from bias tensor B, or
// zeros when B is nil.
func RecurrentBiases[T FloatType](B *tensor.Tensor, nGates, hiddenSize int) (wb, rb []T) {
	size := nGates * hiddenSize

	if B == nil {
		zeros := make([]T, size)

		return zeros, zeros
	}

	biases := tensor.Data[T](B)

	return biases[:size], biases[size : 2*size]
}

// InitialState returns a copy of the initial state of a recurrent operator, or zeros when
// the initial state is nil.
func InitialState[T FloatType](initial *tensor.Tensor, size int) []T {
	state := make([]T, size)
	if initial != nil {
		copy(state, tensor.Data[T](initial))
	}

	return state
}
