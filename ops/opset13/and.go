package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinAndInputs = 2
	MaxAndInputs = 2
)

// And represents the ONNX and operator.
type And struct{}

// newAnd creates a new and operator.
func newAnd() ops.Operator {
	return &And{}
}

// Init initializes the and operator.
func (a *And) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the and operator.
func (a *And) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := ops.ApplyBooleanOperator(
		in1,
		in2,
		func(a, b bool) bool { return a && b },
	)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *And) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *And) GetMinInputs() int {
	return MinAndInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *And) GetMaxInputs() int {
	return MaxAndInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *And) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *And) String() string {
	return "and operator"
}
