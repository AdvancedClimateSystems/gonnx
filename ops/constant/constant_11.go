package constant

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Constant11 represents the ONNX constant operator.
type Constant11 struct {
	ops.BaseOperator

	value tensor.Tensor
}

// newConstant11 creates a new constant operator.
func newConstant11() ops.Operator {
	return &Constant11{
		BaseOperator: ops.NewBaseOperator(
			11,
			0,
			0,
			[][]tensor.Dtype{},
			"constant",
		),
	}
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
