package opset13

import (
	"fmt"
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Asinh represents the ONNX asinh operator.
type Asinh struct{}

// newAsinh creates a new asinh operator.
func newAsinh() ops.Operator {
	return &Asinh{}
}

// Init initializes the asinh operator.
func (a *Asinh) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

type AsinhDType interface {
	float32 | float64
}

// Apply applies the asinh operator.
func (a *Asinh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var out tensor.Tensor
	var err error
	if inputs[0].Dtype() == tensor.Float32 {
		out, err = inputs[0].Apply(asinh[float32])
	} else if inputs[0].Dtype() == tensor.Float64 {
		out, err = inputs[0].Apply(asinh[float64])
	} else {
		return nil, fmt.Errorf(ops.UnsupportedDtypeErrTemplate, inputs[0].Dtype(), a)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *Asinh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *Asinh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *Asinh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *Asinh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *Asinh) String() string {
	return "asinh operator"
}

func asinh[T AsinhDType](x T) T {
	return T(math.Asinh(float64(x)))
}
