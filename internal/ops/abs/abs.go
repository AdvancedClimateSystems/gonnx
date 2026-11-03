package abs

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var absTypeConstraint = [][]tensor.DataType{
	{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Abs represents the ONNX abs operator.
type Abs struct {
	ops.BaseOperator
}

// newAbs creates a new abs operator.
func newAbs(version int, typeConstraint [][]tensor.DataType) ops.Operator {
	return &Abs{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"abs",
		),
	}
}

// Init initializes the abs operator.
func (a *Abs) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the abs operator.
func (a *Abs) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], abs[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], abs[float64])
	case tensor.Int8:
		out = ops.Map(inputs[0], abs[int8])
	case tensor.Int16:
		out = ops.Map(inputs[0], abs[int16])
	case tensor.Int32:
		out = ops.Map(inputs[0], abs[int32])
	case tensor.Int64:
		out = ops.Map(inputs[0], abs[int64])
	case tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64:
		out = inputs[0]
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), a.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func abs[T ops.NumericType](x T) T {
	if x < 0 {
		return -x
	}

	return x
}
