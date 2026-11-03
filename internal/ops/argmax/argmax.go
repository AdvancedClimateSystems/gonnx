package argmax

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var argMaxTypeConstraints = [][]tensor.DataType{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// ArgMax represents the ONNX argmax operator.
type ArgMax struct {
	ops.BaseOperator

	axis            int
	keepDims        bool
	selectLastIndex bool
}

// newArgMax creates a new argmax operator.
func newArgMax(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &ArgMax{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"argmax",
		),
		keepDims:        true,
		selectLastIndex: false,
	}
}

// Init initializes the argmax operator.
func (a *ArgMax) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case "axis":
			a.axis = int(attr.GetI())
		case "keepdims":
			a.keepDims = ops.Int64ToBool(attr.GetI())
		case "select_last_index":
			a.selectLastIndex = ops.Int64ToBool(attr.GetI())

			// We have no way yet to perform argmax and keeping the
			// last index as max in case of duplicates, so if this
			// attribute is true, we raise an unsupported error.
			if a.selectLastIndex {
				return ops.ErrUnsupportedAttribute(attr.GetName(), a)
			}
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), a)
		}
	}

	return nil
}

// Apply applies the argmax operator.
func (a *ArgMax) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.ArgMax(inputs[0], a.axis, a.keepDims)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
