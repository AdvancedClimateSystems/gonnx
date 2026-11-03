package cosh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var coshTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Cosh represents the ONNX cosh operator.
type Cosh struct {
	ops.BaseOperator
}

// newCosh creates a new cosh operator.
func newCosh(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (c *Cosh) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], cosh[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], cosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), c.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func cosh[T ops.FloatType](x T) T {
	return T(math.Cosh(float64(x)))
}
