package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinDivInputs = 2
	MaxDivInputs = 2
)

// Div represents the ONNX div operator.
type Div struct{}

// newDiv creates a new div operator.
func newDiv() ops.Operator {
	return &Div{}
}

// Init initializes the div operator.
func (d *Div) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the div operator.
func (d *Div) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Div,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (d *Div) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(d, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (d *Div) GetMinInputs() int {
	return MinDivInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (d *Div) GetMaxInputs() int {
	return MaxDivInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (d *Div) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (d *Div) String() string {
	return "div operator"
}
