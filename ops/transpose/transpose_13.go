package transpose

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinTranspose13Inputs = 1
	MaxTranspose13Inputs = 1
)

// Transpose13 represents the ONNX transpose operator.
type Transpose13 struct {
	perm []int
}

// newTranspose13 creates a new transpose operator.
func newTranspose13() ops.Operator {
	return &Transpose13{}
}

// Init initializes the transpose operator.
func (t *Transpose13) Init(n *onnx.NodeProto) error {
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
func (t *Transpose13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Transpose(inputs[0], t.perm...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (t *Transpose13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Transpose13) GetMinInputs() int {
	return MinTranspose13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Transpose13) GetMaxInputs() int {
	return MaxTranspose13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (t *Transpose13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (t *Transpose13) String() string {
	return "transpose13 operator"
}
