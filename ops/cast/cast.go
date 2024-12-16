package cast

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var castTypeConstraints = [][]tensor.Dtype{
	{tensor.Int16, tensor.Uint16, tensor.Int32, tensor.Uint32, tensor.Int64, tensor.Uint64, tensor.Float32, tensor.Float64},
}

// Cast represents the ONNX cast operator.
type Cast struct {
	ops.BaseOperator

	to int32 // DataType to cast to, as defined by TensorProto
}

// newCast creates a new cast operator.
func newCast(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Cast{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"cast",
		),
	}
}

// Init initializes the cast operator.
func (c *Cast) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]
	if attr.GetName() == "to" {
		c.to = int32(attr.GetI())
	} else {
		return ops.ErrInvalidAttribute(attr.GetName(), c)
	}

	return nil
}

// Apply applies the cast operator.
func (c *Cast) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.ConvertTensorDtype(inputs[0], c.to)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
