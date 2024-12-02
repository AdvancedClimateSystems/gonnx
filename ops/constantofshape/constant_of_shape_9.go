package constantofshape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinConstantOfShape9Inputs = 1
	MaxConstantOfShape9Inputs = 1
)

// ConstantOfShape9 represents the ONNX constant of shape operator.
type ConstantOfShape9 struct {
	// One element tensor, giving the value and type of the output tensor
	// defaults to value 0 and type float32.
	value *tensor.Dense
}

// newConstantOfShape9 creates a new constant of shape operator.
func newConstantOfShape9() ops.Operator {
	return &ConstantOfShape9{}
}

// Init initializes the constant of shape operator.
func (c *ConstantOfShape9) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) > 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	if len(attributes) == 1 {
		attr := attributes[0]
		if attr.GetName() == "value" {
			t, err := onnx.TensorFromProto(attr.GetT())
			if err != nil {
				return err
			}

			c.value = tensor.New(tensor.WithBacking(t.Data()))
			if c.value.Len() != 1 {
				return ops.ErrInvalidTensor("expected tensor to have one element", c)
			}
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), c)
		}
	} else {
		c.value = tensor.New(tensor.FromScalar(float32(0.0)))
	}

	return nil
}

// Apply applies the constant of shape operator.
func (c *ConstantOfShape9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	shape, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[0].Data()))
	if err != nil {
		return nil, err
	}

	// Empty dimensions in a tensor are not supported
	for i := range shape {
		if shape[i] <= 0 {
			return nil, ops.ErrInvalidTensor("empty dimensions are not allowed", c)
		}
	}

	t := tensor.New(tensor.WithShape(shape...), tensor.Of(c.value.Dtype()))

	t, err = t.AddScalar(c.value, true)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{t}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *ConstantOfShape9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *ConstantOfShape9) GetMinInputs() int {
	return MinConstantOfShape9Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *ConstantOfShape9) GetMaxInputs() int {
	return MaxConstantOfShape9Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *ConstantOfShape9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *ConstantOfShape9) String() string {
	return "constantofshape9 operator"
}
