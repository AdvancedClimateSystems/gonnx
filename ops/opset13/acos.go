package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Acos represents the ONNX acos operator.
type Acos struct{}

// newAcos creates a new acos operator.
func newAcos() ops.Operator {
	return &Acos{}
}

// Init initializes the acos operator.
func (c *Acos) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the acos operator.
func (c *Acos) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var out tensor.Tensor

	var err error

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(acos[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(acos[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Acos) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Acos) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Acos) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Acos) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Acos) String() string {
	return "acos operator"
}

func acos[T ops.FloatType](x T) T {
	return T(math.Acos(float64(x)))
}
