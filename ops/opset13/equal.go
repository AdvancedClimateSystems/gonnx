package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Equal represents the ONNX equal operator.
type Equal struct{}

// newEqual creates a new equal operator.
func newEqual() ops.Operator {
	return &Equal{}
}

// Init initializes the equal operator.
func (e *Equal) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := tensor.ElEq(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (e *Equal) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(e, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (e *Equal) GetMinInputs() int {
	return 2
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (e *Equal) GetMaxInputs() int {
	return 2
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
