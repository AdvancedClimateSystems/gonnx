package equal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinEqual7Inputs = 2
	MaxEqual7Inputs = 2
)

// Equal7 represents the ONNX equal operator.
type Equal7 struct{}

// newEqual7 creates a new equal operator.
func newEqual7() ops.Operator {
	return &Equal7{}
}

// Init initializes the equal operator.
func (e *Equal7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Equal,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (e *Equal7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(e, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (e *Equal7) GetMinInputs() int {
	return MinEqual7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (e *Equal7) GetMaxInputs() int {
	return MaxEqual7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (e *Equal7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool, tensor.Int32, tensor.Int64}, {tensor.Bool, tensor.Int32, tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (e *Equal7) String() string {
	return "equal7 operator"
}
