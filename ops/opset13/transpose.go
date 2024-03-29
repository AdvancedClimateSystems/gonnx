package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinTransposeInputs = 1
	MaxTransposeInputs = 1
)

// Transpose represents the ONNX transpose operator.
type Transpose struct {
	perm []int
}

// newTranspose creates a new transpose operator.
func newTranspose() ops.Operator {
	return &Transpose{}
}

// Init initializes the transpose operator.
func (t *Transpose) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), t)
	}

	attr := attributes[0]

	if attr.GetName() != "perm" {
		return ops.ErrInvalidAttribute(attr.GetName(), t)
	}

	attrPerm := attr.GetInts()
	for _, val := range attrPerm {
		t.perm = append(t.perm, int(val))
	}

	return nil
}

// Apply applies the transpose operator.
func (t *Transpose) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Transpose(inputs[0], t.perm...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (t *Transpose) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Transpose) GetMinInputs() int {
	return MinTransposeInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Transpose) GetMaxInputs() int {
	return MaxTransposeInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (t *Transpose) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (t *Transpose) String() string {
	return "transpose operator"
}
