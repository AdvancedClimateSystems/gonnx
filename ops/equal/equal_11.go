package equal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinEqual11Inputs = 2
	MaxEqual11Inputs = 2
)

// Equal11 represents the ONNX equal operator.
type Equal11 struct{}

// newEqual11 creates a new equal operator.
func NewEqual11() ops.Operator {
	return &Equal11{}
}

// Init initializes the equal operator.
func (e *Equal11) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Equal,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (e *Equal11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(e, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (e *Equal11) GetMinInputs() int {
	return MinEqual11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (e *Equal11) GetMaxInputs() int {
	return MaxEqual11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (e *Equal11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (e *Equal11) String() string {
	return "equal11 operator"
}
