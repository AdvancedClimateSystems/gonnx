package sigmoid

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Sigmoid6 represents the ONNX sigmoid operator.
type Sigmoid6 struct{}

// newSigmoid6 returns a new sigmoid operator.
func newSigmoid6() ops.Operator {
	return &Sigmoid6{}
}

// Init initializes the sigmoid operator.
func (s *Sigmoid6) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (s *Sigmoid6) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Sigmoid(inputs[0])

	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sigmoid6) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sigmoid6) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sigmoid6) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sigmoid6) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sigmoid6) String() string {
	return "sigmoid6 operator"
}
