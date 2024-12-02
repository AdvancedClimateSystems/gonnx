package reducemax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinReduceMax1Attributes = 1
	MaxReduceMax1Attributes = 2
)

// ReduceMax1 represents the ONNX reduceMax operator.
type ReduceMax1 struct {
	axes     []int
	keepDims bool
}

// newReduceMax1 creates a new reduceMax operator.
func newReduceMax1() ops.Operator {
	return &ReduceMax1{
		axes:     []int{},
		keepDims: true,
	}
}

// Init initializes the reduceMax operator.
func (r *ReduceMax1) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) == 0 || len(attributes) > MaxReduceMax1Attributes {
		return ops.ErrInvalidOptionalAttributeCount(MinReduceMax1Attributes, MaxReduceMax1Attributes, len(attributes), r)
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
func (r *ReduceMax1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (r *ReduceMax1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *ReduceMax1) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *ReduceMax1) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *ReduceMax1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *ReduceMax1) String() string {
	return "reducemax1 operator"
}
