package concat

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var concatTypeConstraints = [][]tensor.DataType{ops.AllTypes}

const (
	MinConcatInputs = 1
)

// Concat represents the ONNX concat operator.
type Concat struct {
	ops.BaseOperator

	axis int
}

// newConcat creates a new concat operator.
func newConcat(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (c *Concat) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) == 1 {
		return inputs, nil
	}

	out, err := ops.Concat(c.axis, inputs...)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
// Because Concat can have an infinite number of inputs, we set the maximum number
// of inputs dynamically, based on our inputs. Every input can have any type.
func (c *Concat) ValidateInputs(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	inputTypeConstraints := make([][]tensor.DataType, len(inputs))

	for i := 0; i < len(inputs); i++ {
		inputTypeConstraints[i] = ops.AllTypes
	}

	operator := ops.NewBaseOperator(
		c.Version(),
		c.GetMinInputs(),
		len(inputs),
		inputTypeConstraints,
		"concat",
	)

	return ops.ValidateInputs(operator, inputs)
}
