package acos

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Acos represents the ONNX acos operator.
type Acos struct {
	ops.BaseOperator
}

// newAcos creates a new acos operator.
func newAcos() ops.Operator {
	return &Acos{
		BaseOperator: ops.NewBaseOperator(
			7,
			1,
			1,
			[][]tensor.DataType{{tensor.Float32, tensor.Float64}},
			"acos",
		),
	}
}

// Init initializes the acos operator.
func (c *Acos) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acos operator.
func (c *Acos) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], acos[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], acos[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), c.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func acos[T ops.FloatType](x T) T {
	return T(math.Acos(float64(x)))
}
