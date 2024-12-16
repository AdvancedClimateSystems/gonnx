package sin

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var sinTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Sin represents the ONNX sin operator.
type Sin struct {
	ops.BaseOperator
}

// newSin creates a new sin operator.
func newSin(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (s *Sin) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(sin[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(sin[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), s.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func sin[T ops.FloatType](x T) T {
	return T(math.Sin(float64(x)))
}
