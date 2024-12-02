package greater

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreater9Inputs = 2
	MaxGreater9Inputs = 2
)

// Greater9 represents the ONNX greater operator.
type Greater9 struct{}

// newGreater9 creates a new greater operator.
func newGreater9() ops.Operator {
	return &Greater9{}
}

// Init initializes the greater operator.
func (g *Greater9) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greater operator.
func (g *Greater9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Greater9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Greater9) GetMinInputs() int {
	return MinGreater9Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Greater9) GetMaxInputs() int {
	return MaxGreater9Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Greater9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Greater9) String() string {
	return "greater9 operator"
}
