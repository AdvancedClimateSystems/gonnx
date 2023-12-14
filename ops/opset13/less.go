package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLessInputs = 2
	MaxLessInputs = 2
)

// Less represents the ONNX less operator.
type Less struct{}

// newLess creates a new less operator.
func newLess() ops.Operator {
	return &Less{}
}

// Init initializes the less operator.
func (l *Less) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the less operator.
func (l *Less) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *Less) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *Less) GetMinInputs() int {
	return MinLessInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *Less) GetMaxInputs() int {
	return MaxLessInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *Less) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *Less) String() string {
	return "less operator"
}
