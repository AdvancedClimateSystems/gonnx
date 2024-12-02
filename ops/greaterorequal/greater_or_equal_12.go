package greaterorequal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreaterOrEqual12Inputs = 2
	MaxGreaterOrEqual12Inputs = 2
)

// GreaterOrEqual12 represents the ONNX greaterOrEqual operator.
type GreaterOrEqual12 struct{}

// newGreaterOrEqual12 creates a new greaterOrEqual operator.
func newGreaterOrEqual12() ops.Operator {
	return &GreaterOrEqual12{}
}

// Init initializes the greaterOrEqual operator.
func (g *GreaterOrEqual12) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greaterOrEqual operator.
func (g *GreaterOrEqual12) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gte,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *GreaterOrEqual12) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *GreaterOrEqual12) GetMinInputs() int {
	return MinGreaterOrEqual12Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *GreaterOrEqual12) GetMaxInputs() int {
	return MaxGreaterOrEqual12Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *GreaterOrEqual12) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *GreaterOrEqual12) String() string {
	return "greaterOrEqual12 operator"
}
