package cast

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinCast13Inputs = 1
	MaxCast13Inputs = 1
)

// Cast13 represents the ONNX cast operator.
type Cast13 struct {
	to int32 // DataType to cast to, as defined by TensorProto
}

// newCast13 creates a new cast operator.
func NewCast13() ops.Operator {
	return &Cast13{}
}

// Init initializes the cast operator.
func (c *Cast13) Init(n *onnx.NodeProto) error {
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
func (c *Cast13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.ConvertTensorDtype(inputs[0], c.to)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Cast13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Cast13) GetMinInputs() int {
	return MinCast13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Cast13) GetMaxInputs() int {
	return MaxCast13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Cast13) GetInputTypeConstraints() [][]tensor.Dtype {
	// tensor.String is specified by ONNX but not supported here yet.
	return [][]tensor.Dtype{
		{
			tensor.Int16, tensor.Uint16, tensor.Int32, tensor.Uint32, tensor.Int64, tensor.Uint64,
			tensor.Float32, tensor.Float64,
		},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Cast13) String() string {
	return "cast13 operator"
}
