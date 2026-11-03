package acosh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Acosh represents the ONNX acosh operator.
type Acosh struct {
	ops.BaseOperator
}

// newAcosh creates a new acosh operator.
func newAcosh() ops.Operator {
	return &Acosh{
		BaseOperator: ops.NewBaseOperator(9, 1, 1, [][]tensor.DataType{{tensor.Float32, tensor.Float64}}, "acosh"),
	}
}

// Init initializes the acosh operator.
func (c *Acosh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acosh operator.
func (c *Acosh) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], acosh[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], acosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), c.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func acosh[T ops.FloatType](x T) T {
	return T(math.Acosh(float64(x)))
}
