package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinArgMaxInputs = 1
	MaxArgMaxInputs = 1
)

// ArgMax represents the ONNX argmax operator.
type ArgMax struct {
	axis            int
	keepDims        bool
	selectLastIndex bool
}

// newArgMax creates a new argmax operator.
func newArgMax() ops.Operator {
	return &ArgMax{
		keepDims:        true,
		selectLastIndex: false,
	}
}

type ArgMaxAttribute string

const (
	axis            = "axis"
	keepDims        = "keepdims"
	selectLastIndex = "select_last_index"
)

// Init initializes the argmax operator.
func (a *ArgMax) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case axis:
			a.axis = int(attr.GetI())
		case keepDims:
			a.keepDims = ops.Int64ToBool(attr.GetI())
		case selectLastIndex:
			a.selectLastIndex = ops.Int64ToBool(attr.GetI())

			// We have no way yet to perform argmax and keeping the
			// last index as max in case of duplicates, so if this
			// attribute is true, we raise an unsupported error.
			if a.selectLastIndex {
				return ops.ErrUnsupportedAttribute(attr.GetName(), a)
			}
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), a)
		}
	}

	return nil
}

// Apply applies the argmax operator.
func (a *ArgMax) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis := ops.ConvertNegativeAxis(a.axis, len(inputs[0].Shape()))

	reduced, err := tensor.Argmax(inputs[0], axis)
	if err != nil {
		return nil, err
	}

	// Keep the reduced dimension, i.e. if the reduced axis was '1', and
	// the original shape was (2, 4, 5), the reduced shape would be (2, 5).
	// If keepDims is true, that shape should be (2, 1, 5).
	if a.keepDims {
		newShape := inputs[0].Shape()
		newShape[axis] = 1

		if err := reduced.Reshape(newShape...); err != nil {
			return nil, err
		}
	}

	// The tensor.Argmax function returns data of type int, but according to
	// the ONNX standard this operator should return int64.
	backing, ok := reduced.Data().([]int)
	if !ok {
		return nil, ops.ErrTypeAssert("int", reduced.Dtype())
	}

	backing2 := make([]int64, len(backing))
	for i := range backing {
		backing2[i] = int64(backing[i])
	}

	reduced = tensor.New(tensor.WithShape(reduced.Shape()...), tensor.WithBacking(backing2))

	return []tensor.Tensor{reduced}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *ArgMax) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *ArgMax) GetMinInputs() int {
	return MinArgMaxInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *ArgMax) GetMaxInputs() int {
	return MaxArgMaxInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *ArgMax) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *ArgMax) String() string {
	return "argmax operator"
}
