package reducemean

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var reduceMeanTypeConstraints = [][]tensor.DataType{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// ReduceMean represents the ONNX reduceMean operator.
type ReduceMean struct {
	ops.BaseOperator

	axes     []int
	keepDims bool
}

// newReduceMean creates a new reduceMean operator.
func newReduceMean(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &ReduceMean{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"reducemean",
		),
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMin operator.
func (r *ReduceMean) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			r.axes = axes
		case "keepdims":
			r.keepDims = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the reduceMean operator.
func (r *ReduceMean) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.Reduce(inputs[0], ops.ReduceMean, r.axes, r.keepDims)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
