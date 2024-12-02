package less

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinLess7Inputs = 2
	MaxLess7Inputs = 2
)

// Less7 represents the ONNX less operator.
type Less7 struct{}

// newLess7 creates a new less operator.
func newLess7() ops.Operator {
	return &Less7{}
}

// Init initializes the less operator.
func (l *Less7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the less operator.
func (l *Less7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *Less7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *Less7) GetMinInputs() int {
	return MinLess7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *Less7) GetMaxInputs() int {
	return MaxLess7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *Less7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *Less7) String() string {
	return "less7 operator"
}
