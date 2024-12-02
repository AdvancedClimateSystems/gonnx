package acosh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Acosh9 represents the ONNX acosh operator.
type Acosh9 struct{}

// newAcosh9 creates a new acosh operator.
func newAcosh9() ops.Operator {
	return &Acosh9{}
}

// Init initializes the acosh operator.
func (c *Acosh9) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acosh operator.
func (c *Acosh9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(acosh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(acosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Acosh9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Acosh9) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Acosh9) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Acosh9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Acosh9) String() string {
	return "acosh9 operator"
}

func acosh[T ops.FloatType](x T) T {
	return T(math.Acosh(float64(x)))
}
