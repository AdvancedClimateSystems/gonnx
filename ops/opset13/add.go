package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAddInputs = 2
	MaxAddInputs = 2
)

// Add represents the ONNX add operator.
type Add struct{}

// newAdd creates a new add operator.
func newAdd() ops.Operator {
	return &Add{}
}

// Init initializes the add operator.
func (a *Add) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the add operator.
func (a *Add) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Add,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Add) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Add) GetMinInputs() int {
	return MinAddInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Add) GetMaxInputs() int {
	return MaxAddInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Add) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Add) String() string {
	return "add operator"
}
