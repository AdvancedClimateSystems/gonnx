package tan

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var tanTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Tan represents the ONNX tan operator.
type Tan struct {
	ops.BaseOperator
}

// newTan creates a new tan operator.
func newTan(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (t *Tan) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(tan[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(tan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), t.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func tan[T ops.FloatType](x T) T {
	return T(math.Tan(float64(x)))
}
