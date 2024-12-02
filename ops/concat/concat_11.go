package concat

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinConcat11Inputs = 1
)

// Concat11 represents the ONNX concat operator.
type Concat11 struct {
	axis                 int
	maxInputs            int
	inputTypeConstraints [][]tensor.Dtype
}

// newConcat11 creates a new concat operator.
func newConcat11() ops.Operator {
	return &Concat11{}
}

// Init initializes the concat operator.
func (c *Concat11) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	c.axis = int(attributes[0].GetI())

	return nil
}

// Apply applies the concat operator.
func (c *Concat11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Not sure why this is possible, but minimum number of inputs is said to be 1.
	if len(inputs) == 1 {
		return inputs, nil
	}

	axis := c.axis
	if axis < 0 {
		axis = len(inputs[0].Shape()) + axis
	}

	out, err := tensor.Concat(axis, inputs[0], inputs[1:]...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Concat11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Because Concat11 can have an infinite number of inputs, we set the maximum number
	// of inputs dynamically, based on our inputs. Every input can have any type.
	c.maxInputs = len(inputs)
	c.inputTypeConstraints = make([][]tensor.Dtype, len(inputs))

	for i := 0; i < len(inputs); i++ {
		c.inputTypeConstraints[i] = ops.AllTypes
	}

	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Concat11) GetMinInputs() int {
	return MinConcat11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Concat11) GetMaxInputs() int {
	return c.maxInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Concat11) GetInputTypeConstraints() [][]tensor.Dtype {
	return c.inputTypeConstraints
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Concat11) String() string {
	return "concat11 operator"
}
