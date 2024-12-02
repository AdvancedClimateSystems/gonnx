package and

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinAnd7Inputs = 2
	MaxAnd7Inputs = 2
)

// And7 represents the ONNX and operator.
type And7 struct{}

// newAnd7 creates a new and operator.
func newAnd7() ops.Operator {
	return &And7{}
}

// Init initializes the and operator.
func (a *And7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the and operator.
func (a *And7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.And,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *And7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *And7) GetMinInputs() int {
	return MinAnd7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *And7) GetMaxInputs() int {
	return MaxAnd7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *And7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *And7) String() string {
	return "and7 operator"
}
