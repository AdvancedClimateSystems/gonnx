package opset13

import (
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/onnx"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"
	"gorgonia.org/tensor"
)

// Tanh represents the tanh operator.
type Tanh struct{}

// newTanh returns a new tanh operator.
func newTanh() ops.Operator {
	return &Tanh{}
}

// Init initializes the sigmoid operator.
func (t *Tanh) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (t *Tanh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Tanh(inputs[0])
	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (t *Tanh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Tanh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Tanh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list with for every input tensor a list of allowed types.
func (t *Tanh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String returns a small name of the operator that can be used in formatting errors or logs.
func (t *Tanh) String() string {
	return "tanh operator"
}
