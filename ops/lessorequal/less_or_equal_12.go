package lessorequal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLessOrEqual12Inputs = 2
	MaxLessOrEqual12Inputs = 2
)

// LessOrEqual12 represents the ONNX lessOrEqual operator.
type LessOrEqual12 struct{}

// newLessOrEqual12 creates a new lessOrEqual operator.
func newLessOrEqual12() ops.Operator {
	return &LessOrEqual12{}
}

// Init initializes the lessOrEqual operator.
func (l *LessOrEqual12) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the lessOrEqual operator.
func (l *LessOrEqual12) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lte,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LessOrEqual12) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LessOrEqual12) GetMinInputs() int {
	return MinLessOrEqual12Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LessOrEqual12) GetMaxInputs() int {
	return MaxLessOrEqual12Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LessOrEqual12) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LessOrEqual12) String() string {
	return "lessorequal12 operator"
}
