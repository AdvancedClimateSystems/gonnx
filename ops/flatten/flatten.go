package flatten

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Flatten provides common functionality for all Flatten versions.
type Flatten struct {
	ops.BaseOperator
	axis int
}

func newFlatten(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Flatten{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"flatten",
		),
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
func (f *Flatten) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	inputShape := inputs[0].Shape()
	rank := len(inputShape)

	axis := f.axis
	if axis < 0 {
		axis = rank + axis
	}

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	var err error
	// Handle the special case where axis is 0.
	if axis == 0 {
		err = out.Reshape(1, ops.NElements(inputShape...))
	} else {
		err = out.Reshape(ops.NElements(inputShape[:axis]...), ops.NElements(inputShape[axis:]...))
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
