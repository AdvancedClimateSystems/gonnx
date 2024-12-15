package constantofshape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var constantOfShapeTypeConstraints = [][]tensor.Dtype{{tensor.Int64}}

// ConstantOfShape represents the ONNX constant of shape operator.
type ConstantOfShape struct {
	ops.BaseOperator

	// One element tensor, giving the value and type of the output tensor
	// defaults to value 0 and type float32.
	value *tensor.Dense
}

// newConstantOfShape creates a new constant of shape operator.
func newConstantOfShape(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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

			c.value = tensor.New(tensor.WithBacking(t.Data()))
			if c.value.Len() != 1 {
				return ops.ErrInvalidTensor("expected tensor to have one element", c)
			}
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), c)
		}
	} else {
		c.value = tensor.New(tensor.FromScalar(float32(0.0)))
	}

	return nil
}

// Apply applies the constant of shape operator.
func (c *ConstantOfShape) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	shape, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[0].Data()))
	if err != nil {
		return nil, err
	}

	// Empty dimensions in a tensor are not supported
	for i := range shape {
		if shape[i] <= 0 {
			return nil, ops.ErrInvalidTensor("empty dimensions are not allowed", c)
		}
	}

	t := tensor.New(tensor.WithShape(shape...), tensor.Of(c.value.Dtype()))

	t, err = t.AddScalar(c.value, true)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{t}, err
}
