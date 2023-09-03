package opset13

import (
	"fmt"
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Sin represents the ONNX sin operator.
type Sin struct{}

// newSin creates a new sin operator.
func newSin() ops.Operator {
	return &Sin{}
}

// Init initializes the sin operator.
func (s *Sin) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

type SinDType interface {
	float32 | float64
}

// Apply applies the sin operator.
func (s *Sin) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var out tensor.Tensor
	var err error
	if inputs[0].Dtype() == tensor.Float32 {
		out, err = inputs[0].Apply(sin[float32])
	} else if inputs[0].Dtype() == tensor.Float64 {
		out, err = inputs[0].Apply(sin[float64])
	} else {
		return nil, fmt.Errorf(ops.UnsupportedDtypeErrTemplate, inputs[0].Dtype(), s)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Sin) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Sin) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Sin) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Sin) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Sin) String() string {
	return "sin operator"
}

func sin[T SinDType](x T) T {
	return T(math.Sin(float64(x)))
}
