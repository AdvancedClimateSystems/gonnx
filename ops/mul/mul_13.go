package mul

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinMul13Inputs = 2
	MaxMul13Inputs = 2
)

// Mul13 represents the ONNX mul operator.
type Mul13 struct{}

// newMul13 creates a new mul operator.
func newMul13() ops.Operator {
	return &Mul13{}
}

// Init initializes the mul operator.
func (m *Mul13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the mul operator.
func (m *Mul13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Mul,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (m *Mul13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(m, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (m *Mul13) GetMinInputs() int {
	return MinMul13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (m *Mul13) GetMaxInputs() int {
	return MaxMul13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (m *Mul13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (m *Mul13) String() string {
	return "mul13 operator"
}
