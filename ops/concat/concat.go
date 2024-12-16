package concat

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var concatTypeConstraints = [][]tensor.Dtype{ops.AllTypes}

const (
	MinConcatInputs = 1
)

// Concat represents the ONNX concat operator.
type Concat struct {
	ops.BaseOperator

	axis int
}

// newConcat creates a new concat operator.
func newConcat(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Concat{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"concat",
		),
	}
}

// Init initializes the concat operator.
func (c *Concat) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
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
// Because Concat can have an infinite number of inputs, we set the maximum number
// of inputs dynamically, based on our inputs. Every input can have any type.
func (c *Concat) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	inputTypeConstraints := make([][]tensor.Dtype, len(inputs))

	for i := 0; i < len(inputs); i++ {
		inputTypeConstraints[i] = ops.AllTypes
	}

	c.BaseOperator = ops.NewBaseOperator(
		c.BaseOperator.Version(),
		c.BaseOperator.GetMinInputs(),
		len(inputs),
		inputTypeConstraints,
		"concat",
	)

	return ops.ValidateInputs(c.BaseOperator, inputs)
}
