package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinEqualInputs = 2
	MaxEqualInputs = 2
)

// Equal represents the ONNX equal operator.
type Equal struct{}

// newEqual creates a new equal operator.
func newEqual() ops.Operator {
	return &Equal{}
}

// Init initializes the equal operator.
func (e *Equal) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Equal,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (e *Equal) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(e, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (e *Equal) GetMinInputs() int {
	return MinEqualInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (e *Equal) GetMaxInputs() int {
	return MaxEqualInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (e *Equal) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (e *Equal) String() string {
	return "equal operator"
}
