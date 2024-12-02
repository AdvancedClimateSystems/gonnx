package or

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinOr7Inputs = 2
	MaxOr7Inputs = 2
)

// Or7 represents the ONNX or operator.
type Or7 struct{}

// newOr7 creates a new or operator.
func newOr7() ops.Operator {
	return &Or7{}
}

// Init initializes the or operator.
func (o *Or7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the or operator.
func (o *Or7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Or,
		ops.MultidirectionalBroadcasting,
	)
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (o *Or7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(o, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (o *Or7) GetMinInputs() int {
	return MinOr7Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (o *Or7) GetMaxInputs() int {
	return MaxOr7Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (o *Or7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (o *Or7) String() string {
	return "or7 operator"
}
