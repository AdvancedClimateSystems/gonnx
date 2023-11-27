package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Atan represents the ONNX atan operator.
type Atan struct{}

// newAtan creates a new atan operator.
func newAtan() ops.Operator {
	return &Atan{}
}

// Init initializes the atan operator.
func (a *Atan) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the atan operator.
func (a *Atan) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(atan[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(atan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Atan) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Atan) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Atan) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Atan) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Atan) String() string {
	return "atan operator"
}

func atan[T ops.FloatType](x T) T {
	return T(math.Atan(float64(x)))
}
