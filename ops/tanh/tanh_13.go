package tanh

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Tanh13 represents the tanh operator.
type Tanh13 struct{}

// newTanh13 returns a new tanh operator.
func newTanh13() ops.Operator {
	return &Tanh13{}
}

// Init initializes the sigmoid operator.
func (t *Tanh13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (t *Tanh13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Tanh(inputs[0])

	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (t *Tanh13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Tanh13) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Tanh13) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list with for every input tensor a list of allowed types.
func (t *Tanh13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String returns a small name of the operator that can be used in formatting errors or logs.
func (t *Tanh13) String() string {
	return "tanh13 operator"
}
