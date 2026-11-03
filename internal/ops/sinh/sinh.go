package sinh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var sinhTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Sinh represents the ONNX sinh operator.
type Sinh struct {
	ops.BaseOperator
}

// newSin creates a new sinh operator.
func newSinh(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Sinh{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"sinh",
		),
	}
}

// Init initializes the sinh operator.
func (s *Sinh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the sinh operator.
func (s *Sinh) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], sinh[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], sinh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), s.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func sinh[T ops.FloatType](x T) T {
	return T(math.Sinh(float64(x)))
}
