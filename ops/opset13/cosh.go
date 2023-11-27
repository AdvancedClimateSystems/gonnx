package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Cosh represents the ONNX cosh operator.
type Cosh struct{}

// newCosh creates a new cosh operator.
func newCosh() ops.Operator {
	return &Cosh{}
}

// Init initializes the cosh operator.
func (c *Cosh) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the cosh operator.
func (c *Cosh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(cosh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(cosh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), c)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Cosh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Cosh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Cosh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Cosh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Cosh) String() string {
	return "cosh operator"
}

func cosh[T ops.FloatType](x T) T {
	return T(math.Cosh(float64(x)))
}
