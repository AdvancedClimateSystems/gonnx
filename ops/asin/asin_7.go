package asin

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Asin7 represents the ONNX asin operator.
type Asin7 struct{}

// newSin creates a new asin operator.
func NewAsin7() ops.Operator {
	return &Asin7{}
}

// Init initializes the asin operator.
func (s *Asin7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the asin operator.
func (s *Asin7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(asin[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(asin[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), s)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Asin7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Asin7) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Asin7) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Asin7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Asin7) String() string {
	return "asin7 operator"
}

func asin[T ops.FloatType](x T) T {
	return T(math.Asin(float64(x)))
}
