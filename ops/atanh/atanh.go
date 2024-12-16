package atanh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var atanhTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Atanh represents the ONNX atanh operator.
type Atanh struct {
	ops.BaseOperator
}

// newAtanh creates a new atanh operator.
func newAtanh(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Atanh{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"atanh",
		),
	}
}

// Init initializes the atanh operator.
func (a *Atanh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the atanh operator.
func (a *Atanh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(atanh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(atanh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func atanh[T ops.FloatType](x T) T {
	return T(math.Atanh(float64(x)))
}
