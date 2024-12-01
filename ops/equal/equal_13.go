package equal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinEqual13Inputs = 2
	MaxEqual13Inputs = 2
)

// Equal13 represents the ONNX equal operator.
type Equal13 struct{}

// newEqual13 creates a new equal operator.
func NewEqual13() ops.Operator {
	return &Equal13{}
}

// Init initializes the equal operator.
func (e *Equal13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Equal,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (e *Equal13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(e, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (e *Equal13) GetMinInputs() int {
	return MinEqual13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (e *Equal13) GetMaxInputs() int {
	return MaxEqual13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (e *Equal13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (e *Equal13) String() string {
	return "equal13 operator"
}
