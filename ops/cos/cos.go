package cos

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var cosTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Cos represents the ONNX cos operator.
type Cos struct {
	ops.BaseOperator
}

// newCos creates a new cos operator.
func newCos(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (c *Cos) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(cos[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(cos[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func cos[T ops.FloatType](x T) T {
	return T(math.Cos(float64(x)))
}
