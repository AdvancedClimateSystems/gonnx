package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLessOrEqualInputs = 2
	MaxLessOrEqualInputs = 2
)

// LessOrEqual represents the ONNX lessOrEqual operator.
type LessOrEqual struct{}

// newLessOrEqual creates a new lessOrEqual operator.
func newLessOrEqual() ops.Operator {
	return &LessOrEqual{}
}

// Init initializes the lessOrEqual operator.
func (l *LessOrEqual) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the lessOrEqual operator.
func (l *LessOrEqual) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lte,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LessOrEqual) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LessOrEqual) GetMinInputs() int {
	return MinLessOrEqualInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LessOrEqual) GetMaxInputs() int {
	return MaxLessOrEqualInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LessOrEqual) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LessOrEqual) String() string {
	return "lessOrEqual operator"
}
