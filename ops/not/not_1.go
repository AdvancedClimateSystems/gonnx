package not

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Not1 represents the ONNX not operator.
type Not1 struct{}

// newNot1 creates a new not operator.
func newNot1() ops.Operator {
	return &Not1{}
}

// Init initializes the not operator.
func (n *Not1) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the not operator.
func (n *Not1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := inputs[0].Apply(not)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (n *Not1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(n, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (n *Not1) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (n *Not1) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (n *Not1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (n *Not1) String() string {
	return "not1 operator"
}

func not(x bool) bool {
	return !x
}
