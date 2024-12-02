package sub

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSub7Inputs = 2
	MaxSub7Inputs = 2
)

// Sub7 represents the ONNX sub operator.
type Sub7 struct{}

// newSub7 creates a new sub operator.
func newSub7() ops.Operator {
	return &Sub7{}
}

// Init initializes the sub operator.
func (s *Sub7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the sub operator.
func (s *Sub7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Sub,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sub7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sub7) GetMinInputs() int {
	return MinSub7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sub7) GetMaxInputs() int {
	return MaxSub7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sub7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sub7) String() string {
	return "sub7 operator"
}
