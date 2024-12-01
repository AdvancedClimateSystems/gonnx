package div

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinDiv13Inputs = 2
	MaxDiv13Inputs = 2
)

// Div13 represents the ONNX div operator.
type Div13 struct{}

// newDiv13 creates a new div operator.
func NewDiv13() ops.Operator {
	return &Div13{}
}

// Init initializes the div operator.
func (d *Div13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the div operator.
func (d *Div13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Div,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (d *Div13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(d, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (d *Div13) GetMinInputs() int {
	return MinDiv13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (d *Div13) GetMaxInputs() int {
	return MaxDiv13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (d *Div13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (d *Div13) String() string {
	return "div13 operator"
}
