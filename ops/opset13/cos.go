package opset13

import (
	"fmt"
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Cos represents the ONNX cos operator.
type Cos struct{}

// newCos creates a new cos operator.
func newCos() ops.Operator {
	return &Cos{}
}

// Init initializes the cos operator.
func (c *Cos) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

type CosDType interface {
	float32 | float64
}

// Apply applies the sin operator.
func (c *Cos) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var out tensor.Tensor
	var err error
	if inputs[0].Dtype() == tensor.Float32 {
		out, err = inputs[0].Apply(cos[float32])
	} else if inputs[0].Dtype() == tensor.Float64 {
		out, err = inputs[0].Apply(cos[float64])
	} else {
		return nil, fmt.Errorf(ops.UnsupportedDtypeErrTemplate, inputs[0].Dtype(), s)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Cos) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Cos) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Cos) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Cos) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Cos) String() string {
	return "cos operator"
}

func cos[T CosDType](x T) T {
	return T(math.Cos(float64(x)))
}
