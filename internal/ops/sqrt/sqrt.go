package sqrt

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var sqrtTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Sqrt represents the ONNX sqrt operator.
type Sqrt struct {
	ops.BaseOperator
}

// newSqrt creates a new sqrt operator.
func newSqrt(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Sqrt{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"sqrt",
		),
	}
}

// Init initializes the sqrt operator.
func (s *Sqrt) Init(_ *onnx.NodeProto) error {
	return nil
}

// Apply applies the sqrt operator.
func (s *Sqrt) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], sqrt[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], sqrt[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), s.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func sqrt[T ops.FloatType](x T) T {
	return T(math.Sqrt(float64(x)))
}
