package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinCastInputs = 1
	MaxCastInputs = 1
)

// Cast represents the ONNX cast operator.
type Cast struct {
	to int32 // DataType to cast to, as defined by TensorProto
}

// newCast creates a new cast operator.
func newCast() ops.Operator {
	return &Cast{}
}

// Init initializes the cast operator.
func (c *Cast) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), c)
	}

	attr := attributes[0]
	if attr.GetName() == "to" {
		c.to = int32(attr.GetI())
	} else {
		return ops.ErrInvalidAttribute(attr.GetName(), c)
	}

	return nil
}

// Apply applies the cast operator.
func (c *Cast) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.ConvertTensorDtype(inputs[0], c.to)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Cast) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Cast) GetMinInputs() int {
	return MinCastInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Cast) GetMaxInputs() int {
	return MaxCastInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Cast) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{
			tensor.Int16, tensor.Uint16, tensor.Int32, tensor.Uint32, tensor.Int64, tensor.Uint64,
			tensor.Float32, tensor.Float64,
		},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Cast) String() string {
	return "cast operator"
}
