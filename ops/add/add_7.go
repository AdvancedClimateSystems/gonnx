package add

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinAdd7Inputs = 2
	MaxAdd7Inputs = 2
)

// Add7 represents the ONNX add operator.
type Add7 struct{}

// newAdd7 creates a new add operator.
func NewAdd7() ops.Operator {
	return &Add7{}
}

// Init initializes the add operator.
func (a *Add7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the add operator.
func (a *Add7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Add,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Add7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Add7) GetMinInputs() int {
	return MinAdd7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Add7) GetMaxInputs() int {
	return MaxAdd7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Add7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Add7) String() string {
	return "add7 operator"
}
