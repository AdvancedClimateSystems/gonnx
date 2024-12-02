package add

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAdd13Inputs = 2
	MaxAdd13Inputs = 2
)

// Add13 represents the ONNX add operator.
type Add13 struct{}

// newAdd13 creates a new add operator.
func newAdd13() ops.Operator {
	return &Add13{}
}

// Init initializes the add operator.
func (a *Add13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the add operator.
func (a *Add13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Add,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Add13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Add13) GetMinInputs() int {
	return MinAdd13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Add13) GetMaxInputs() int {
	return MaxAdd13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Add13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Add13) String() string {
	return "add13 operator"
}
