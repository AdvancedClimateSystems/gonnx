package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Constant11 represents the ONNX constant operator.
type Constant11 struct {
	value tensor.Tensor
}

// newConstant11 creates a new constant operator.
func newConstant11() ops.Operator {
	return &Constant11{}
}

// Init initializes the constant operator. It supports all constant types except
// `sparse_value`.
func (c *Constant11) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]

	switch attr.GetName() {
	case sparseValue:
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	case value:
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
func (c *Constant11) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Constant11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Constant11) GetMinInputs() int {
	return 0
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Constant11) GetMaxInputs() int {
	return 0
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Constant11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Constant11) String() string {
	return "constant11 operator"
}
