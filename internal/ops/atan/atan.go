package atan

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var atanTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// Atan represents the ONNX atan operator.
type Atan struct {
	ops.BaseOperator
}

// newAtan creates a new atan operator.
func newAtan(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (a *Atan) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var out *tensor.Tensor

	switch inputs[0].DataType() {
	case tensor.Float32:
		out = ops.Map(inputs[0], atan[float32])
	case tensor.Float64:
		out = ops.Map(inputs[0], atan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].DataType().String(), a.BaseOperator)
	}

	return []*tensor.Tensor{out}, nil
}

func atan[T ops.FloatType](x T) T {
	return T(math.Atan(float64(x)))
}
