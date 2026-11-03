package asinh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var asinhTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Asinh represents the ONNX asinh operator.
type Asinh struct {
	ops.BaseOperator
}

// newAsinh creates a new asinh operator.
func newAsinh(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Asinh{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"asinh",
		),
	}
}

// Init initializes the asinh operator.
func (a *Asinh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the asinh operator.
func (a *Asinh) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], asinh[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], asinh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), a.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func asinh[T ops.FloatType](x T) T {
	return T(math.Asinh(float64(x)))
}
