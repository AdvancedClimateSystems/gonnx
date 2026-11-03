package gather

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var gatherTypeConstraints = [][]tensor.DataType{
	ops.AllTypes,
	{tensor.Int32, tensor.Int64},
}

// Gather represents the ONNX gather operator.
type Gather struct {
	ops.BaseOperator

	axis int // axis to gather on, default is 0
}

// newGather creates a new gather operator.
func newGather(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Gather{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"gather",
		),
		axis: 0,
	}
}

// Init initializes the gather operator.
func (g *Gather) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) == 1 {
		attr := attributes[0]

		if attr.GetName() == axis {
			g.axis = int(attr.GetI())
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	} else if len(attributes) > 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), g)
	}

	return nil
}

// Apply applies the gather operator.
func (g *Gather) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	// Convert the indices (of data type Int32 or Int64) to ints.
	indicesData, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	data := inputs[0]

	// Make sure axis is in the correct range (according to the size of the data tensor)
	rank := len(data.Shape())
	dataAxis := g.axis

	if dataAxis < -rank || dataAxis > rank-1 {
		return nil, ops.ErrAxisOutOfRange(rank, rank, dataAxis)
	}
	// Offset axis if a negative index is given.
	if dataAxis < 0 {
		dataAxis += rank
	}

	// Make sure the input indices are all in the correct range (according to the size of the
	// dimension which is selected by `axis`)
	axisDimSize := data.Shape()[dataAxis]
	if !ops.AllInRange(indicesData, -axisDimSize, axisDimSize-1) {
		return nil, ops.ErrNotAllAxesInRange(axisDimSize, axisDimSize)
	}

	ops.OffsetArrayIfNegative(indicesData, axisDimSize)

	output, err := ops.Gather(data, indicesData, inputs[1].Shape(), dataAxis)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{output}, nil
}
