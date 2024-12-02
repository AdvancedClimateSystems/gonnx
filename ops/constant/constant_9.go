package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Constant9 represents the ONNX constant operator.
type Constant9 struct {
	value tensor.Tensor
}

// newConstant9 creates a new constant operator.
func newConstant9() ops.Operator {
	return &Constant9{}
}

// Init initializes the constant operator.
func (c *Constant9) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]

	switch attr.GetName() {
	case "value":
		t, err := onnx.TensorFromProto(attr.GetT())
		if err != nil {
			return err
		}

		c.value = t
	default:
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	}

	return nil
}

// Apply applies the constant operator.
func (c *Constant9) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Constant9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Constant9) GetMinInputs() int {
	return 0
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Constant9) GetMaxInputs() int {
	return 0
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Constant9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Constant9) String() string {
	return "constant9 operator"
}
