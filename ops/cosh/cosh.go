package cosh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var coshTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Cosh represents the ONNX cosh operator.
type Cosh struct {
	ops.BaseOperator
}

// newCosh creates a new cosh operator.
func newCosh(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Cosh{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"cosh",
		),
	}
}

// Init initializes the cosh operator.
func (c *Cosh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the cosh operator.
func (c *Cosh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(cosh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(cosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func cosh[T ops.FloatType](x T) T {
	return T(math.Cosh(float64(x)))
}
