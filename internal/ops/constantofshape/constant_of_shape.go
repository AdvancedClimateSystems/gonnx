package constantofshape

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var constantOfShapeTypeConstraints = [][]tensor.DataType{{tensor.Int64}}

// ConstantOfShape represents the ONNX constant of shape operator.
type ConstantOfShape struct {
	ops.BaseOperator

	// One element tensor, giving the value and type of the output tensor
	// defaults to value 0 and type float32.
	value *tensor.Tensor
}

// newConstantOfShape creates a new constant of shape operator.
func newConstantOfShape(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &ConstantOfShape{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"constantofshape",
		),
	}
}

// Init initializes the constant of shape operator.
func (c *ConstantOfShape) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) > 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	if len(attributes) == 1 {
		attr := attributes[0]
		if attr.GetName() == "value" {
			t, err := onnx.TensorFromProto(attr.GetT())
			if err != nil {
				return err
			}

			if t.Size() != 1 {
				return ops.ErrInvalidTensor("expected tensor to have one element", c)
			}

			c.value, err = t.Reshape()
			if err != nil {
				return err
			}
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), c)
		}
	} else {
		c.value = tensor.Scalar(float32(0.0))
	}

	return nil
}

// Apply applies the constant of shape operator.
func (c *ConstantOfShape) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	shape, err := ops.AnyToIntSlice(inputs[0].Data())
	if err != nil {
		return nil, err
	}

	// Empty dimensions in a tensor are not supported
	for i := range shape {
		if shape[i] <= 0 {
			return nil, ops.ErrInvalidTensor("empty dimensions are not allowed", c)
		}
	}

	t, err := ops.BroadcastTo(c.value, shape)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{t}, nil
}
