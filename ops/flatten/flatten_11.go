package flatten

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinFlatten11Inputs = 1
	MaxFlatten11Inputs = 1
)

// Flatten11 represents the ONNX flatten operator.
type Flatten11 struct {
	axis int
}

// newFlatten11 creates a new flatten operator.
func NewFlatten11() ops.Operator {
	return &Flatten11{
		axis: 1,
	}
}

// Init initializes the flatten operator.
func (f *Flatten11) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axis":
			f.axis = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), f)
		}
	}

	return nil
}

// Apply applies the flatten operator.
func (f *Flatten11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
	// In the special case where axis is 0, we reshape the tensor to shape
	// (1, <n_elements>). This is ONNX defined behaviour.
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (f *Flatten11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(f, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (f *Flatten11) GetMinInputs() int {
	return MinFlatten11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (f *Flatten11) GetMaxInputs() int {
	return MaxFlatten11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (f *Flatten11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (f *Flatten11) String() string {
	return "flatten11 operator"
}
