package opset13

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Tan represents the ONNX tan operator.
type Tan struct{}

// newTan creates a new tan operator.
func newTan() ops.Operator {
	return &Tan{}
}

// Init initializes the tan operator.
func (t *Tan) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the tan operator.
func (t *Tan) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var (
		out tensor.Tensor
		err error
	)

	switch inputs[0].Dtype() {
	case tensor.Float32:
		out, err = inputs[0].Apply(tan[float32])
	case tensor.Float64:
		out, err = inputs[0].Apply(tan[float64])
	default:
		return nil, ops.ErrInvalidInputType(0, inputs[0].Dtype().String(), t)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (t *Tan) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Tan) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Tan) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (t *Tan) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (t *Tan) String() string {
	return "tan operator"
}

func tan[T ops.FloatType](x T) T {
	return T(math.Tan(float64(x)))
}
