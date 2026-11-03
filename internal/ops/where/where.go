package where

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var whereTypeConstraints = [][]tensor.DataType{
	{tensor.Bool},
	ops.AllTypes,
	ops.AllTypes,
}

// Where represents the ONNX where operator.
type Where struct {
	ops.BaseOperator
}

// newWhere creates a new where operator.
func newWhere(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Where{
		BaseOperator: ops.NewBaseOperator(
			version,
			3,
			3,
			typeConstraints,
			"where",
		),
	}
}

// Init initializes the where operator.
func (w *Where) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the where operator.
func (w *Where) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	condition, X, Y := inputs[0], inputs[1], inputs[2]
	if X.DataType() != Y.DataType() {
		return nil, ops.ErrInvalidInputType(2, Y.DataType().String(), w.BaseOperator)
	}

	shape, err := ops.MultidirectionalBroadcastShape(X.Shape(), Y.Shape())
	if err != nil {
		return nil, err
	}

	shape, err = ops.MultidirectionalBroadcastShape(condition.Shape(), shape)
	if err != nil {
		return nil, err
	}

	broadcast := make([]*tensor.Tensor, 3)

	for i, t := range []*tensor.Tensor{condition, X, Y} {
		broadcast[i], err = ops.BroadcastTo(t, shape)
		if err != nil {
			return nil, err
		}
	}

	conditionValues := tensor.Data[bool](broadcast[0])

	var out *tensor.Tensor

	switch X.DataType() {
	case tensor.Float32:
		out = where[float32](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Float64:
		out = where[float64](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Int8:
		out = where[int8](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Int16:
		out = where[int16](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Int32:
		out = where[int32](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Int64:
		out = where[int64](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Uint8:
		out = where[uint8](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Uint16:
		out = where[uint16](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Uint32:
		out = where[uint32](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Uint64:
		out = where[uint64](conditionValues, broadcast[1], broadcast[2], shape)
	case tensor.Bool:
		out = where[bool](conditionValues, broadcast[1], broadcast[2], shape)
	default:
		return nil, ops.ErrInvalidInputType(1, X.DataType().String(), w.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func where[T tensor.Element](condition []bool, X, Y *tensor.Tensor, shape tensor.Shape) *tensor.Tensor {
	x, y := tensor.Data[T](X), tensor.Data[T](Y)
	out := make([]T, len(condition))

	for i, takeX := range condition {
		if takeX {
			out[i] = x[i]
		} else {
			out[i] = y[i]
		}
	}

	return tensor.New(out, shape...)
}
