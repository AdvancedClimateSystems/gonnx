package relu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Relu6 represents the ONNX relu operator.
type Relu6 struct{}

// newRelu6 creates a new relu operator.
func newRelu6() ops.Operator {
	return &Relu6{}
}

// Init initializes the relu operator.
func (r *Relu6) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the relu operator.
func (r *Relu6) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.ReLU(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *Relu6) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *Relu6) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *Relu6) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *Relu6) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *Relu6) String() string {
	return "relu6 operator"
}
