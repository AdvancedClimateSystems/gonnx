package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Constant12 represents the ONNX constant operator.
type Constant12 struct {
	value tensor.Tensor
}

// newConstant12 creates a new constant operator.
func newConstant12() ops.Operator {
	return &Constant12{}
}

// Init initializes the constant operator. It supports all constant types except
// `sparse_value`, `value_string`, and `value_strings`.
func (c *Constant12) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]

	switch attr.GetName() {
	case sparseValue, valueString, valueStrings:
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	case value:
		t, err := onnx.TensorFromProto(attr.GetT())
		if err != nil {
			return err
		}

		c.value = t
	case valueFloat:
		c.value = tensor.New(tensor.FromScalar(attr.GetF()))
	case valueFloats:
		floats := attr.GetFloats()
		c.value = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
	case valueInt:
		c.value = tensor.New(tensor.FromScalar(attr.GetI()))
	case valueInts:
		ints := attr.GetInts()
		c.value = tensor.New(tensor.WithShape(len(ints)), tensor.WithBacking(ints))
	default:
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	}

	return nil
}

// Apply applies the constant operator.
func (c *Constant12) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Constant12) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Constant12) GetMinInputs() int {
	return 0
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Constant12) GetMaxInputs() int {
	return 0
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Constant12) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Constant12) String() string {
	return "constant12 operator"
}
