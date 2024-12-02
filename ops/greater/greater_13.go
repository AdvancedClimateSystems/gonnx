package greater

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreater13Inputs = 2
	MaxGreater13Inputs = 2
)

// Greater13 represents the ONNX greater operator.
type Greater13 struct{}

// newGreater13 creates a new greater operator.
func newGreater13() ops.Operator {
	return &Greater13{}
}

// Init initializes the greater operator.
func (g *Greater13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greater operator.
func (g *Greater13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Greater13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Greater13) GetMinInputs() int {
	return MinGreater13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Greater13) GetMaxInputs() int {
	return MaxGreater13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Greater13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Greater13) String() string {
	return "greater13 operator"
}
