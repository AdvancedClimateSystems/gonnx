package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreaterOrEqualInputs = 2
	MaxGreaterOrEqualInputs = 2
)

// GreaterOrEqual represents the ONNX greaterOrEqual operator.
type GreaterOrEqual struct{}

// newGreaterOrEqual creates a new greaterOrEqual operator.
func newGreaterOrEqual() ops.Operator {
	return &GreaterOrEqual{}
}

// Init initializes the greaterOrEqual operator.
func (g *GreaterOrEqual) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greaterOrEqual operator.
func (g *GreaterOrEqual) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gte,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *GreaterOrEqual) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *GreaterOrEqual) GetMinInputs() int {
	return MinGreaterOrEqualInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *GreaterOrEqual) GetMaxInputs() int {
	return MaxGreaterOrEqualInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *GreaterOrEqual) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *GreaterOrEqual) String() string {
	return "greaterOrEqual operator"
}
