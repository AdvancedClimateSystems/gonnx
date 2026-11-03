package expand

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var expandTypeConstraints = [][]tensor.DataType{ops.AllTypes, {tensor.Int64}}

// Expand represents the ONNX expand operator.
type Expand struct {
	ops.BaseOperator
}

// newExpand creates a new expand operator.
func newExpand(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Expand{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"expand",
		),
	}
}

// Init initializes the expand operator.
func (f *Expand) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the expand operator.
func (f *Expand) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	shape, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	outShape, err := ops.MultidirectionalBroadcastShape(inputs[0].Shape(), shape)
	if err != nil {
		return nil, err
	}

	out, err := ops.BroadcastTo(inputs[0], outShape)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
