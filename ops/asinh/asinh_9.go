package asinh

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Asinh9 represents the ONNX asinh operator.
type Asinh9 struct{}

// newAsinh9 creates a new asinh operator.
func newAsinh9() ops.Operator {
	return &Asinh9{}
}

// Init initializes the asinh operator.
func (a *Asinh9) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the asinh operator.
func (a *Asinh9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(asinh[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(asinh[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), a)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Asinh9) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Asinh9) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Asinh9) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Asinh9) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Asinh9) String() string {
	return "asinh9 operator"
}

func asinh[T ops.FloatType](x T) T {
	return T(math.Asinh(float64(x)))
}
