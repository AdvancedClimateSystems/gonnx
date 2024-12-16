package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Unsqueeze11 represents version 11 of the ONNX unsqueeze operator.
type Unsqueeze11 struct {
	ops.BaseOperator

	axes []int
}

// newUnsqueeze11 creates a new unsqueeze operator.
func newUnsqueeze11() ops.Operator {
	return &Unsqueeze11{
		BaseOperator: ops.NewBaseOperator(
			11,
			1,
			1,
			[][]tensor.Dtype{ops.AllTypes},
			"unsqueeze",
		),
	}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze11) Init(n *onnx.NodeProto) error {
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
func (u *Unsqueeze11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	dataShape := inputs[0].Shape()

	outputRank := len(dataShape) + len(u.axes)

	if !ops.AllInRange(u.axes, -outputRank, outputRank-1) {
		return nil, ops.ErrNotAllAxesInRange(outputRank, outputRank)
	}

	// negative entries should be offset by the rank of the output tensor
	// i.e. -1 -> outputRank - 1, -outputrank -> 0
	ops.OffsetArrayIfNegative(u.axes, outputRank)

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
