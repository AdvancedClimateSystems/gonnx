package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Atanh represents the ONNX atanh operator.
type Atanh struct{}

// newAtanh creates a new atanh operator.
func newAtanh() ops.Operator {
	return &Atanh{}
}

// Init initializes the atanh operator.
func (a *Atanh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the atanh operator.
func (a *Atanh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(atanh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(atanh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Atanh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Atanh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Atanh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Atanh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Atanh) String() string {
	return "atanh operator"
}

func atanh[T ops.FloatType](x T) T {
	return T(math.Atanh(float64(x)))
}
