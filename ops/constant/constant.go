package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Constant represents the ONNX constant operator.
type Constant struct {
	ops.BaseOperator

	value tensor.Tensor
}

// newConstant creates a new constant operator.
func newConstant(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Constant{
		BaseOperator: ops.NewBaseOperator(
			version,
			0,
			0,
			typeConstraints,
			"constant",
		),
	}
}

// Init initializes the constant operator. It supports all constant types except
// `sparse_value`, `value_string`, and `value_strings`.
func (c *Constant) Init(n *onnx.NodeProto) error {
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
func (c *Constant) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}
