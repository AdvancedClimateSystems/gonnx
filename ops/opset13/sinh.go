package opset13

import (
	"fmt"
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Sinh represents the ONNX sinh operator.
type Sinh struct{}

// newSin creates a new sinh operator.
func newSinh() ops.Operator {
	return &Sinh{}
}

// Init initializes the sinh operator.
func (s *Sinh) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

type SinhDType interface {
	float32 | float64
}

// Apply applies the sinh operator.
func (s *Sinh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var out tensor.Tensor
	var err error
	if inputs[0].Dtype() == tensor.Float32 {
		out, err = inputs[0].Apply(sinh[float32])
	} else if inputs[0].Dtype() == tensor.Float64 {
		out, err = inputs[0].Apply(sinh[float64])
	} else {
		return nil, fmt.Errorf(ops.UnsupportedDtypeErrTemplate, inputs[0].Dtype(), s)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sinh) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sinh) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sinh) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sinh) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sinh) String() string {
	return "sinh operator"
}

func sinh[T SinhDType](x T) T {
	return T(math.Sinh(float64(x)))
}
