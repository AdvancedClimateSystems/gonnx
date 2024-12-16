package atan

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var atanTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Atan represents the ONNX atan operator.
type Atan struct {
	ops.BaseOperator
}

// newAtan creates a new atan operator.
func newAtan(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Atan{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"atan",
		),
	}
}

// Init initializes the atan operator.
func (a *Atan) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the atan operator.
func (a *Atan) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(atan[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(atan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func atan[T ops.FloatType](x T) T {
	return T(math.Atan(float64(x)))
}
