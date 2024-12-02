package greater

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreater7Inputs = 2
	MaxGreater7Inputs = 2
)

// Greater7 represents the ONNX greater operator.
type Greater7 struct{}

// newGreater7 creates a new greater operator.
func newGreater7() ops.Operator {
	return &Greater7{}
}

// Init initializes the greater operator.
func (g *Greater7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greater operator.
func (g *Greater7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gt,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Greater7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Greater7) GetMinInputs() int {
	return MinGreater7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Greater7) GetMaxInputs() int {
	return MaxGreater7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Greater7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Greater7) String() string {
	return "greater7 operator"
}
