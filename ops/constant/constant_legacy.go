package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// ConstantLegacy represents the ONNX constant operator for version 9 and 1.
type ConstantLegacy struct {
	ops.BaseOperator

	value tensor.Tensor
}

// newConstantLegacy creates a new constant operator.
func newConstantLegacy(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ConstantLegacy{
		BaseOperator: ops.NewBaseOperator(
			version,
			0,
			0,
			typeConstraints,
			"constant",
		),
	}
}

// Init initializes the constant operator.
func (c *ConstantLegacy) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]

	switch attr.GetName() {
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
func (c *ConstantLegacy) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}
