package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Relu represents the ONNX relu operator.
type Relu struct{}

// newRelu creates a new relu operator.
func newRelu() ops.Operator {
	return &Relu{}
}

// Init initializes the relu operator.
func (r *Relu) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the relu operator.
func (r *Relu) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]

	typedZero, err := ops.GetValueAsTensorType(0.0, t.Dtype())
	if err != nil {
		return nil, err
	}

	comparison, err := tensor.Gt(t, typedZero, tensor.AsSameType())
	if err != nil {
		return nil, err
	}

	out, err := tensor.Mul(t, comparison)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *Relu) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *Relu) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *Relu) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *Relu) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *Relu) String() string {
	return "relu operator"
}
