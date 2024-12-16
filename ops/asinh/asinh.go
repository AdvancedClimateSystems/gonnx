package asinh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var asinhTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Asinh represents the ONNX asinh operator.
type Asinh struct {
	ops.BaseOperator
}

// newAsinh creates a new asinh operator.
func newAsinh(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (a *Asinh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(asinh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(asinh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func asinh[T ops.FloatType](x T) T {
	return T(math.Asinh(float64(x)))
}
