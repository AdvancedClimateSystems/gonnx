package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSubInputs = 2
	MaxSubInputs = 2
)

// Sub represents the ONNX sub operator.
type Sub struct{}

// newSub creates a new sub operator.
func newSub() ops.Operator {
	return &Sub{}
}

// Init initializes the sub operator.
func (s *Sub) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the sub operator.
func (s *Sub) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Sub,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sub) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sub) GetMinInputs() int {
	return MinSubInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sub) GetMaxInputs() int {
	return MaxSubInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sub) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sub) String() string {
	return "sub operator"
}
