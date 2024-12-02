package flatten

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// FlattenBase provides common functionality for all Flatten versions.
type FlattenBase struct {
	version              int
	axis                 int
	minInputs            int
	maxInputs            int
	inputTypeConstraints [][]tensor.Dtype
}

// Init initializes the flatten operator.
func (f *FlattenBase) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case axis:
			f.axis = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), f)
		}
	}

	return nil
}

// Apply applies the flatten operator.
func (f *FlattenBase) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	inputShape := inputs[0].Shape()
	rank := len(inputShape)

	axis := f.axis
	if axis < 0 {
		axis = rank + axis
	}

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	var err error
	// Handle the special case where axis is 0.
	if axis == 0 {
		err = out.Reshape(1, ops.NElements(inputShape...))
	} else {
		err = out.Reshape(ops.NElements(inputShape[:axis]...), ops.NElements(inputShape[axis:]...))
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs for the operator.
func (f *FlattenBase) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(f, inputs)
}

// GetMinInputs returns the minimum number of input tensors.
func (f *FlattenBase) GetMinInputs() int {
	return f.minInputs
}

// GetMaxInputs returns the maximum number of input tensors.
func (f *FlattenBase) GetMaxInputs() int {
	return f.maxInputs
}

// GetInputTypeConstraints returns allowed input types.
func (f *FlattenBase) GetInputTypeConstraints() [][]tensor.Dtype {
	return f.inputTypeConstraints
}

func (f *FlattenBase) String() string {
	return fmt.Sprintf("flatten<%d> operator", f.version)
}
