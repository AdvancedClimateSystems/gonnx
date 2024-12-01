package div

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinDiv7Inputs = 2
	MaxDiv7Inputs = 2
)

// Div7 represents the ONNX div operator.
type Div7 struct{}

// newDiv7 creates a new div operator.
func NewDiv7() ops.Operator {
	return &Div7{}
}

// Init initializes the div operator.
func (d *Div7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the div operator.
func (d *Div7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Div,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (d *Div7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(d, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (d *Div7) GetMinInputs() int {
	return MinDiv7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (d *Div7) GetMaxInputs() int {
	return MaxDiv7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (d *Div7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (d *Div7) String() string {
	return "div7 operator"
}
