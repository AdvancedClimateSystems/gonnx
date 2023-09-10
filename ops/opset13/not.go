package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Not represents the ONNX not operator.
type Not struct{}

// newNot creates a new not operator.
func newNot() ops.Operator {
	return &Not{}
}

// Init initializes the not operator.
func (n *Not) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the not operator.
func (n *Not) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := inputs[0].Apply(not)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (n *Not) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(n, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (n *Not) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (n *Not) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (n *Not) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Bool},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (n *Not) String() string {
	return "not operator"
}

func not(x bool) bool {
	return !x
}
