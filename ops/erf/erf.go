package erf

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var erfTypeConstraints = [][]tensor.Dtype{ops.NumericTypes}

// Erf represents the ONNX erf operator.
type Erf struct {
	ops.BaseOperator
}

// newSin creates a new erf operator.
func newErf(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Erf{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"erf",
		),
	}
}

// Init initializes the erf operator.
func (e *Erf) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the erf operator.
func (e *Erf) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Uint8:
		out, err = inputs[0].Apply(erf[uint8])
	case tensor.Uint16:
		out, err = inputs[0].Apply(erf[uint16])
	case tensor.Uint32:
		out, err = inputs[0].Apply(erf[uint32])
	case tensor.Uint64:
		out, err = inputs[0].Apply(erf[uint64])
	case tensor.Int8:
		out, err = inputs[0].Apply(erf[int8])
	case tensor.Int16:
		out, err = inputs[0].Apply(erf[int16])
	case tensor.Int32:
		out, err = inputs[0].Apply(erf[int32])
	case tensor.Int64:
		out, err = inputs[0].Apply(erf[int64])
	case tensor.Float32:
		out, err = inputs[0].Apply(erf[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(erf[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), e.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func erf[T ops.NumericType](x T) T {
	return T(math.Erf(float64(x)))
}
