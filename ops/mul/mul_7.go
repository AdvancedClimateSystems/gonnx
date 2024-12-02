package mul

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinMul7Inputs = 2
	MaxMul7Inputs = 2
)

// Mul7 represents the ONNX mul operator.
type Mul7 struct{}

// newMul7 creates a new mul operator.
func newMul7() ops.Operator {
	return &Mul7{}
}

// Init initializes the mul operator.
func (m *Mul7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the mul operator.
func (m *Mul7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Mul,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (m *Mul7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(m, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (m *Mul7) GetMinInputs() int {
	return MinMul7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (m *Mul7) GetMaxInputs() int {
	return MaxMul7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (m *Mul7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (m *Mul7) String() string {
	return "mul7 operator"
}
