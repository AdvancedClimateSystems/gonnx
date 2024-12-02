package sub

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSub13Inputs = 2
	MaxSub13Inputs = 2
)

// Sub13 represents the ONNX sub operator.
type Sub13 struct{}

// newSub13 creates a new sub operator.
func newSub13() ops.Operator {
	return &Sub13{}
}

// Init initializes the sub operator.
func (s *Sub13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the sub operator.
func (s *Sub13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Sub,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sub13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sub13) GetMinInputs() int {
	return MinSub13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sub13) GetMaxInputs() int {
	return MaxSub13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sub13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sub13) String() string {
	return "sub13 operator"
}
