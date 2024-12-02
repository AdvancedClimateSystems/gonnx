package abs

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAbs6Inputs = 1
	MaxAbs6Inputs = 1
)

// Abs6 represents the ONNX abs operator.
type Abs6 struct{}

// newAbs6 creates a new abs operator.
func newAbs6() ops.Operator {
	return &Abs6{}
}

// Init initializes the abs operator.
func (a *Abs6) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the abs operator.
func (a *Abs6) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Abs(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Abs6) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Abs6) GetMinInputs() int {
	return MinAbs6Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Abs6) GetMaxInputs() int {
	return MaxAbs6Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Abs6) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Abs6) String() string {
	return "abs6 operator"
}
