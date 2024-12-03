package acos

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
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
			[][]tensor.Dtype{{tensor.Float32, tensor.Float64}},
			"acos",
		),
	}
}

// Init initializes the acos operator.
func (c *Acos) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acos operator.
func (c *Acos) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(acos[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(acos[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func acos[T ops.FloatType](x T) T {
	return T(math.Acos(float64(x)))
}
