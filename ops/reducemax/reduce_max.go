package reducemax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reduceMaxTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

var reduceMax11TypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

const (
	MinReduceMaxAttributes = 1
	MaxReduceMaxAttributes = 2
)

// ReduceMax represents the ONNX reduceMax operator.
type ReduceMax struct {
	ops.BaseOperator

	axes     []int
	keepDims bool
}

// newReduceMax creates a new reduceMax operator.
func newReduceMax(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ReduceMax{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"reducemax",
		),
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMax operator.
func (r *ReduceMax) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) == 0 || len(attributes) > MaxReduceMaxAttributes {
		return ops.ErrInvalidOptionalAttributeCount(MinReduceMaxAttributes, MaxReduceMaxAttributes, len(attributes), r)
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

// Apply applies the reduceMax operator.
func (r *ReduceMax) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	axes := make([]int, len(r.axes))
	for i, axis := range r.axes {
		axes[i] = ops.ConvertNegativeAxis(axis, len(input.Shape()))
	}

	out, err := input.Max(axes...)
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
