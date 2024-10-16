package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinReduceMinAttributes = 1
	MaxReduceMinAttributes = 2
)

// ReduceMin represents the ONNX reduceMin operator.
type ReduceMin struct {
	axes     []int
	keepDims bool
}

// newReduceMin creates a new reduceMin operator.
func newReduceMin() ops.Operator {
	return &ReduceMin{
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

// Apply applies the reduceMin operator.
func (r *ReduceMin) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	axes := make([]int, len(r.axes))
	for i, axis := range r.axes {
		// Convert negative dimensions to positive dimensions as Go does not support
		// negative dimension indexing like Python does.
		if axis < 0 {
			axis = len(input.Shape()) + axis
		}

		axes[i] = axis
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *ReduceMin) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *ReduceMin) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *ReduceMin) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *ReduceMin) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *ReduceMin) String() string {
	return "reduceMin operator"
}
