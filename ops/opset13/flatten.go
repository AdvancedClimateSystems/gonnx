package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinFlattenInputs = 1
	MaxFlattenInputs = 1
)

// Flatten represents the ONNX flatten operator.
type Flatten struct {
	axis int
}

// newFlatten creates a new flatten operator.
func newFlatten() ops.Operator {
	return &Flatten{
		axis: 1,
	}
}

// Init initializes the flatten operator.
func (f *Flatten) Init(n *onnx.NodeProto) error {
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
func (f *Flatten) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (f *Flatten) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(f, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (f *Flatten) GetMinInputs() int {
	return MinFlattenInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (f *Flatten) GetMaxInputs() int {
	return MaxFlattenInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (f *Flatten) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (f *Flatten) String() string {
	return "flatten operator"
}
