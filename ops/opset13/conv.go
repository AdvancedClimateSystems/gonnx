package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

type AutoPadSetting string

const (
	NotSet    AutoPadSetting = "NOTSET"
	SameUpper AutoPadSetting = "SAME_UPPER"
	SameLower AutoPadSetting = "SAME_LOWER"
	Valid     AutoPadSetting = "VALID"
)

// Conv represents the ONNX conv operator.
type Conv struct {
	// Type of padding to apply before doing the convolutions.
	autoPad AutoPadSetting

	// Dilation value along each dimension of the filter.
	dilations []int

	// Numer of groups the input channels and the output channels are divided into.
	group int

	// Shape of the convolutional kernel. Can be present, but if not should be inferred (i.e. useless attribute).
	kernelShape []int

	// Padding for the beginning and ending of each dimension. Cannot be used with autopad setting.
	pads []int

	// Strides along each dimension.
	strides []int
}

// newConv creates a new conv operator.
func newConv() ops.Operator {
	return &Conv{}
}

// Init initializes the conv operator.
func (c *Conv) Init(attributes []*onnx.AttributeProto) error {
	for _, attr := range attributes {
		switch attr.GetName() {
		case "auto_pad":
			c.autoPad = AutoPadSetting(attr.GetS())
		case "linear_before_reset":
			g.linearBeforeReset = ops.Int64ToBool(attr.GetI())
		default:
			return fmt.Errorf(ops.UnsupportedAttrErrTemplate, g, attr.GetName())
		}
	}

	return nil
}

// Apply applies the conv operator.
func (c *Conv) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := tensor.Conv(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Conv) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Conv) GetMinInputs() int {
	return 2
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Conv) GetMaxInputs() int {
	return 3
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Conv) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Conv) String() string {
	return "conv operator"
}
