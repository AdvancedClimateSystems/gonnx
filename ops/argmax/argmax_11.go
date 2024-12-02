package argmax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinArgMax11Inputs = 1
	MaxArgMax11Inputs = 1
)

// ArgMax11 represents the ONNX argmax operator.
type ArgMax11 struct {
	axis            int
	keepDims        bool
	selectLastIndex bool
}

// newArgMax11 creates a new argmax operator.
func newArgMax11() ops.Operator {
	return &ArgMax11{
		keepDims:        true,
		selectLastIndex: false,
	}
}

type ArgMax11Attribute string

// Init initializes the argmax operator.
func (a *ArgMax11) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case axis:
			a.axis = int(attr.GetI())
		case keepDims:
			a.keepDims = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), a)
		}
	}

	return nil
}

// Apply applies the argmax operator.
func (a *ArgMax11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (a *ArgMax11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *ArgMax11) GetMinInputs() int {
	return MinArgMax11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *ArgMax11) GetMaxInputs() int {
	return MaxArgMax11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *ArgMax11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *ArgMax11) String() string {
	return "argmax11 operator"
}
