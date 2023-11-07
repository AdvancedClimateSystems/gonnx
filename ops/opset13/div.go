package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	// MinDivInput is the minimimum amount of inputs the div operator expects.
	MinDivInput = 2

	// MaxDivInput is the maximum amount of inputs the div operator accepts.
	MaxDivInput = 2
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
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := tensor.Div(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (d *Div) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(d, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (d *Div) GetMinInputs() int {
	return MinDivInput
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (d *Div) GetMaxInputs() int {
	return MaxDivInput
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
