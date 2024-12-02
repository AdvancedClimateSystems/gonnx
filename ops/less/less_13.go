package less

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLess13Inputs = 2
	MaxLess13Inputs = 2
)

// Less13 represents the ONNX less operator.
type Less13 struct{}

// newLess13 creates a new less operator.
func newLess13() ops.Operator {
	return &Less13{}
}

// Init initializes the less operator.
func (l *Less13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the less operator.
func (l *Less13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *Less13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *Less13) GetMinInputs() int {
	return MinLess13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *Less13) GetMaxInputs() int {
	return MaxLess13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *Less13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *Less13) String() string {
	return "less13 operator"
}
