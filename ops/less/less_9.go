package less

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLess9Inputs = 2
	MaxLess9Inputs = 2
)

// Less9 represents the ONNX less operator.
type Less9 struct{}

// newLess9 creates a new less operator.
func newLess9() ops.Operator {
	return &Less9{}
}

// Init initializes the less operator.
func (l *Less9) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the less operator.
func (l *Less9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *Less9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *Less9) GetMinInputs() int {
	return MinLess9Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *Less9) GetMaxInputs() int {
	return MaxLess9Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *Less9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *Less9) String() string {
	return "less9 operator"
}
