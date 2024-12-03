package acosh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Acosh represents the ONNX acosh operator.
type Acosh struct {
	ops.BaseOperator
}

// newAcosh creates a new acosh operator.
func newAcosh() ops.Operator {
	return &Acosh{
		BaseOperator: ops.NewBaseOperator(9, 1, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}, "acosh"),
	}
}

// Init initializes the acosh operator.
func (c *Acosh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acosh operator.
func (c *Acosh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(acosh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(acosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func acosh[T ops.FloatType](x T) T {
	return T(math.Acosh(float64(x)))
}
