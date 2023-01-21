package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Add represents the ONNX add operator.
type Add struct{}

// newAdd creates a new add operator.
func newAdd() ops.Operator {
	return &Add{}
}

// Init initializes the add operator.
func (a *Add) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the add operator.
func (a *Add) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := tensor.Add(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Add) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Add) GetMinInputs() int {
	return 2
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Add) GetMaxInputs() int {
	return 2
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Add) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Add) String() string {
	return "add operator"
}
