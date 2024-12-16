package reducemin

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reduceMinTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

var reduceMin11TypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

const (
	MinReduceMinAttributes = 1
	MaxReduceMinAttributes = 2
)

// ReduceMin represents the ONNX reduceMin operator.
type ReduceMin struct {
	ops.BaseOperator

	axes     []int
	keepDims bool
}

// newReduceMin creates a new reduceMin operator.
func newReduceMin(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ReduceMin{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"reducemin",
		),
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMin operator.
func (r *ReduceMin) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) == 0 || len(attributes) > MaxReduceMinAttributes {
		return ops.ErrInvalidOptionalAttributeCount(MinReduceMinAttributes, MaxReduceMinAttributes, len(attributes), r)
	}

	for _, attr := range attributes {
		switch attr.GetName() {
		case axes:
			value, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			r.axes = value
		case keepDims:
			r.keepDims = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the reduceMin operator.
func (r *ReduceMin) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	axes := make([]int, len(r.axes))
	for i, axis := range r.axes {
		axes[i] = ops.ConvertNegativeAxis(axis, len(input.Shape()))
	}

	out, err := input.Min(axes...)
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
