package flatten

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Flatten provides common functionality for all Flatten versions.
type Flatten struct {
	ops.BaseOperator
	axis int
}

func newFlatten(version int, typeConstraint [][]tensor.DataType) ops.Operator {
	return &Flatten{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"flatten",
		),
		axis: 1,
	}
}

// Init initializes the flatten operator.
func (f *Flatten) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case axis:
			f.axis = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), f)
		}
	}

	return nil
}

// Apply applies the flatten operator.
func (f *Flatten) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	inputShape := inputs[0].Shape()
	rank := len(inputShape)

	axis := f.axis
	if axis < 0 {
		axis = rank + axis
	}

	var (
		out *tensor.Tensor
		err error
	)

	if axis == 0 {
		out, err = inputs[0].Reshape(1, ops.NElements(inputShape...))
	} else {
		out, err = inputs[0].Reshape(ops.NElements(inputShape[:axis]...), ops.NElements(inputShape[axis:]...))
	}

	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
