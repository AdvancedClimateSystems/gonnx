package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinReduceMaxAttributes = 1
	MaxReduceMaxAttributes = 2
)

// ReduceMax represents the ONNX reduceMax operator.
type ReduceMax struct {
	axes     []int
	keepDims bool
}

// newReduceMax creates a new reduceMax operator.
func newReduceMax() ops.Operator {
	return &ReduceMax{
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *ReduceMax) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *ReduceMax) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *ReduceMax) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *ReduceMax) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *ReduceMax) String() string {
	return "reduceMax operator"
}
