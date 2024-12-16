package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Unsqueeze1 represents version 1 of the ONNX unsqueeze operator.
type Unsqueeze1 struct {
	ops.BaseOperator

	axes []int
}

// newUnsqueeze1 creates a new unsqueeze operator.
func newUnsqueeze1() ops.Operator {
	return &Unsqueeze1{
		BaseOperator: ops.NewBaseOperator(
			1,
			1,
			1,
			[][]tensor.Dtype{ops.AllTypes},
			"unsqueeze",
		),
	}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze1) Init(n *onnx.NodeProto) error {
	attrs := n.GetAttribute()
	if len(attrs) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attrs), u)
	}

	axes, err := ops.AnyToIntSlice(attrs[0].GetInts())
	if err != nil {
		return err
	}

	u.axes = axes

	return nil
}

// Apply applies the unsqueeze operator.
func (u *Unsqueeze1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	dataShape := inputs[0].Shape()

	outputRank := len(dataShape) + len(u.axes)

	if !ops.AllInRange(u.axes, 0, outputRank-1) {
		return nil, ops.ErrNotAllAxesInRange(outputRank, outputRank)
	}

	sort.Ints(u.axes)

	if ops.HasDuplicates(u.axes) {
		return nil, ops.ErrInvalidInput("axes cannot have duplicate entries after offset", u.BaseOperator)
	}

	newShape := insertOnes(dataShape, u.axes)

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	err := out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
}
