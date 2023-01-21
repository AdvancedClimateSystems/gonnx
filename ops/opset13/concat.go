package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Concat represents the ONNX concat operator.
type Concat struct {
	axis                 int
	maxInputs            int
	inputTypeConstraints [][]tensor.Dtype
}

// newConcat creates a new concat operator.
func newConcat() ops.Operator {
	return &Concat{}
}

// Init initializes the concat operator.
func (c *Concat) Init(attributes []*onnx.AttributeProto) error {
	if len(attributes) != 1 {
		return fmt.Errorf(ops.InvalidAttrCountErrTemplate, c, 1, len(attributes))
	}

	c.axis = int(attributes[0].GetI())
	return nil
}

// Apply applies the concat operator.
func (c *Concat) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (c *Concat) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Because Concat can have an infinite number of inputs, we set the maximum number
	// of inputs dynamically, based on our inputs. Every input can have any type.
	c.maxInputs = len(inputs)
	c.inputTypeConstraints = make([][]tensor.Dtype, len(inputs))
	for i := 0; i < len(inputs); i++ {
		c.inputTypeConstraints[i] = ops.AllTypes
	}

	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Concat) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Concat) GetMaxInputs() int {
	return c.maxInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Concat) GetInputTypeConstraints() [][]tensor.Dtype {
	return c.inputTypeConstraints
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Concat) String() string {
	return "concat operator"
}
