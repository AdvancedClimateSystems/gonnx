package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinGreaterInputs = 2
	MaxGreaterInputs = 2
)

// Greater represents the ONNX greater operator.
type Greater struct{}

// newGreater creates a new greater operator.
func newGreater() ops.Operator {
	return &Greater{}
}

// Init initializes the greater operator.
func (g *Greater) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the greater operator.
func (g *Greater) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := tensor.Gt(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Greater) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Greater) GetMinInputs() int {
	return MinGreaterInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Greater) GetMaxInputs() int {
	return MaxGreaterInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Greater) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Greater) String() string {
	return "greater operator"
}
