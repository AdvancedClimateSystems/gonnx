package tan

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var tanTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Tan represents the ONNX tan operator.
type Tan struct {
	ops.BaseOperator
}

// newTan creates a new tan operator.
func newTan(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Tan{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"tan",
		),
	}
}

// Init initializes the tan operator.
func (t *Tan) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the tan operator.
func (t *Tan) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], tan[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], tan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), t.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func tan[T ops.FloatType](x T) T {
	return T(math.Tan(float64(x)))
}
