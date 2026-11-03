package cos

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var cosTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Cos represents the ONNX cos operator.
type Cos struct {
	ops.BaseOperator
}

// newCos creates a new cos operator.
func newCos(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Cos{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"cos",
		),
	}
}

// Init initializes the cos operator.
func (c *Cos) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the cos operator.
func (c *Cos) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], cos[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], cos[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), c.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func cos[T ops.FloatType](x T) T {
	return T(math.Cos(float64(x)))
}
