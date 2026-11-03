package sin

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var sinTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Sin represents the ONNX sin operator.
type Sin struct {
	ops.BaseOperator
}

// newSin creates a new sin operator.
func newSin(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Sin{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"sin",
		),
	}
}

// Init initializes the sin operator.
func (s *Sin) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the sin operator.
func (s *Sin) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], sin[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], sin[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), s.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func sin[T ops.FloatType](x T) T {
	return T(math.Sin(float64(x)))
}
