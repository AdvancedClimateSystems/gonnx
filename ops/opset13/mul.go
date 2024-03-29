package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinMulInputs = 2
	MaxMulInputs = 2
)

// Mul represents the ONNX mul operator.
type Mul struct{}

// newMul creates a new mul operator.
func newMul() ops.Operator {
	return &Mul{}
}

// Init initializes the mul operator.
func (m *Mul) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the mul operator.
func (m *Mul) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Mul,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (m *Mul) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(m, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (m *Mul) GetMinInputs() int {
	return MinMulInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (m *Mul) GetMaxInputs() int {
	return MaxMulInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (m *Mul) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (m *Mul) String() string {
	return "mul operator"
}
