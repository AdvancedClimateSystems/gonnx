package reducemean

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

var reduceMeanTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// ReduceMean represents the ONNX reduceMean operator.
type ReduceMean struct {
	ops.BaseOperator

	axes     []int
	keepDims bool
}

// newReduceMean creates a new reduceMean operator.
func newReduceMean(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ReduceMean{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"reducemean",
		),
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMin operator.
func (r *ReduceMean) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			r.axes = axes
		case "keepdims":
			r.keepDims = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

func cast(count int, t tensor.Dtype) (any, error) {
	switch t {
	case tensor.Int:
		return count, nil
	case tensor.Int8:
		return int8(count), nil
	case tensor.Int16:
		return int16(count), nil
	case tensor.Int32:
		return int32(count), nil
	case tensor.Int64:
		return int64(count), nil
	case tensor.Uint:
		return uint(count), nil
	case tensor.Uint8:
		return uint8(count), nil
	case tensor.Uint16:
		return uint16(count), nil
	case tensor.Uint32:
		return uint32(count), nil
	case tensor.Uint64:
		return uint64(count), nil
	case tensor.Float32:
		return float32(count), nil
	case tensor.Float64:
		return float64(count), nil
	case tensor.Complex64:
		return complex(float32(count), 0), nil
	case tensor.Complex128:
		return complex(float64(count), 0), nil
	default:
		return nil, errors.Errorf("No methods found for Sum for %v", t)
	}
}

// Apply applies the reduceMean operator.
func (r *ReduceMean) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	if len(r.axes) == 0 {
		axes := make([]int, len(input.Shape()))
		for i := range axes {
			axes[i] = i
		}
	}

	axes := make([]int, len(r.axes))
	for i, axis := range r.axes {
		axes[i] = ops.ConvertNegativeAxis(axis, len(input.Shape()))
	}

	sum, err := input.Sum(axes...)
	if err != nil {
		return nil, err
	}

	// tensor.NonMaskedCount seems to be bugged, so we'll calculate the count manually
	count := 1
	for _, axis := range axes {
		count *= input.Shape()[axis]
	}

	countCast, err := cast(count, sum.Dtype())
	if err != nil {
		return nil, err
	}

	out, err := sum.DivScalar(countCast, true)
	if err != nil {
		return nil, err
	}

	if r.keepDims {
		newShape := input.Shape()
		for _, axes := range axes {
			newShape[axes] = 1
		}

		err := out.Reshape(newShape...)
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, nil
}
