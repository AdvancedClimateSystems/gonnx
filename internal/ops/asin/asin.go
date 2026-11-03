package asin

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var asinTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Asin represents the ONNX asin operator.
type Asin struct {
	ops.BaseOperator
}

// newSin creates a new asin operator.
func newAsin(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Asin{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"asin",
		),
	}
}

// Init initializes the asin operator.
func (s *Asin) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the asin operator.
func (s *Asin) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], asin[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], asin[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), s.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func asin[T ops.FloatType](x T) T {
	return T(math.Asin(float64(x)))
}
