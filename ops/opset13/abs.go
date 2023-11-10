package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAbsInputs = 1
	MaxAbsInputs = 1
)

// Abs represents the ONNX abs operator.
type Abs struct{}

// newAbs creates a new abs operator.
func newAbs() ops.Operator {
	return &Abs{}
}

// Init initializes the abs operator.
func (a *Abs) Init([]*onnx.AttributeProto) error {
	return nil
}

// Apply applies the abs operator.
func (a *Abs) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Abs(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Abs) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Abs) GetMinInputs() int {
	return MinAbsInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Abs) GetMaxInputs() int {
	return MaxAbsInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Abs) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Abs) String() string {
	return "abs operator"
}
