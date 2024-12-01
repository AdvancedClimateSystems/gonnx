package expand

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinExpand8Inputs = 2
	MaxExpand8Inputs = 2
)

// Expand8 represents the ONNX expand operator.
type Expand8 struct{}

// newExpand8 creates a new expand operator.
func NewExpand8() ops.Operator {
	return &Expand8{}
}

// Init initializes the expand operator.
func (f *Expand8) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the expand operator.
func (f *Expand8) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]

	shape, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	// If the new shape has more dimensions than the input tensor, we
	// need to prepend some dimensions to the input tensor shape.
	if len(shape) > len(input.Shape()) {
		input, err = ops.AddExtraDimsToTensor(input, len(shape)-len(input.Shape()))
		if err != nil {
			return nil, err
		}
	}

	for axis := len(shape) - 1; axis >= 0; axis-- {
		if input.Shape()[axis] != shape[axis] {
			input, err = tensor.Repeat(input, axis, shape[axis])
			if err != nil {
				return nil, err
			}
		}
	}

	return []tensor.Tensor{input}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (f *Expand8) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(f, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (f *Expand8) GetMinInputs() int {
	return MinExpand8Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (f *Expand8) GetMaxInputs() int {
	return MaxExpand8Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (f *Expand8) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (f *Expand8) String() string {
	return "expand13 operator"
}
