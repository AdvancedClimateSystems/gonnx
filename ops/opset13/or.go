package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinOrInputs = 2
	MaxOrInputs = 2
)

// Or represents the ONNX or operator.
type Or struct{}

// newOr creates a new or operator.
func newOr() ops.Operator {
	return &Or{}
}

// Init initializes the or operator.
func (o *Or) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the or operator.
func (o *Or) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := ops.ApplyBooleanOperator(
		in1,
		in2,
		func(a, b bool) bool { return a || b },
	)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (o *Or) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(o, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (o *Or) GetMinInputs() int {
	return MinOrInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (o *Or) GetMaxInputs() int {
	return MaxOrInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (o *Or) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (o *Or) String() string {
	return "or operator"
}
