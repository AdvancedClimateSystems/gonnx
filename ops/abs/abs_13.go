package abs

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAbs13Inputs = 1
	MaxAbs13Inputs = 1
)

// Abs13 represents the ONNX abs operator.
type Abs13 struct{}

// newAbs13 creates a new abs operator.
func NewAbs13() ops.Operator {
	return &Abs13{}
}

// Init initializes the abs operator.
func (a *Abs13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the abs operator.
func (a *Abs13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Abs(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Abs13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Abs13) GetMinInputs() int {
	return MinAbs13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Abs13) GetMaxInputs() int {
	return MaxAbs13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Abs13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Abs13) String() string {
	return "abs13 operator"
}
