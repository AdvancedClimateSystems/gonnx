package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Sigmoid represents the ONNX sigmoid operator.
type Sigmoid struct{}

// newSigmoid returns a new sigmoid operator.
func newSigmoid() ops.Operator {
	return &Sigmoid{}
}

// Init initializes the sigmoid operator.
func (s *Sigmoid) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (s *Sigmoid) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Sigmoid(inputs[0])

	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sigmoid) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sigmoid) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sigmoid) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sigmoid) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sigmoid) String() string {
	return "sigmoid operator"
}
