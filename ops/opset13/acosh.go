package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Acosh represents the ONNX acosh operator.
type Acosh struct{}

// newAcosh creates a new acosh operator.
func newAcosh() ops.Operator {
	return &Acosh{}
}

// Init initializes the acosh operator.
func (c *Acosh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the acosh operator.
func (c *Acosh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (c *Acosh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Acosh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Acosh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Acosh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Acosh) String() string {
	return "acosh operator"
}

func acosh[T ops.FloatType](x T) T {
	return T(math.Acosh(float64(x)))
}
