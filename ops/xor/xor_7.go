package xor

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinXor7Inputs = 2
	MaxXor7Inputs = 2
)

// Xor7 represents the ONNX xor operator.
type Xor7 struct{}

// newXor7 creates a new xor operator.
func newXor7() ops.Operator {
	return &Xor7{}
}

// Init initializes the xor operator.
func (x *Xor7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the xor operator.
func (x *Xor7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Xor,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (x *Xor7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(x, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (x *Xor7) GetMinInputs() int {
	return MinXor7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (x *Xor7) GetMaxInputs() int {
	return MaxXor7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (x *Xor7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (x *Xor7) String() string {
	return "xor7 operator"
}
