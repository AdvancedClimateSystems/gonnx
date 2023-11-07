package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	// MinConstInput is the minimimum amount of inputs the constant operator expects.
	MinConstInput = 1

	// MaxConstInput is the maximum amount of inputs the constant operator accepts.
	MaxConstInput = 1
)

// Constant represents the ONNX constant operator.
type Constant struct {
	value tensor.Tensor
}

// newConstant creates a new constant operator.
func newConstant() ops.Operator {
	return &Constant{}
}

// Init initializes the constant operator. It supports all constant types except
// `sparse_value`, `value_string`, and `value_strings`.
func (c *Constant) Init(attributes []*onnx.AttributeProto) error {
	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]

	switch attr.GetName() {
	case "sparse_value", "value_string", "value_strings":
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	case "value":
		t, err := onnx.TensorFromProto(attr.GetT())
		if err != nil {
			return err
		}

		c.value = t
	case "value_float":
		c.value = tensor.New(tensor.FromScalar(attr.GetF()))
	case "value_floats":
		floats := attr.GetFloats()
		c.value = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
	case "value_int":
		c.value = tensor.New(tensor.FromScalar(attr.GetI()))
	case "value_ints":
		ints := attr.GetInts()
		c.value = tensor.New(tensor.WithShape(len(ints)), tensor.WithBacking(ints))
	default:
		return ops.ErrUnsupportedAttribute(attr.GetName(), c)
	}

	return nil
}

// Apply applies the constant operator.
func (c *Constant) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return []tensor.Tensor{c.value}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Constant) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Constant) GetMinInputs() int {
	return 0
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Constant) GetMaxInputs() int {
	return 0
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Constant) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Constant) String() string {
	return "constant operator"
}
