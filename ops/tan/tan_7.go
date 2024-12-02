package tan

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Tan7 represents the ONNX tan operator.
type Tan7 struct{}

// newTan7 creates a new tan operator.
func newTan7() ops.Operator {
	return &Tan7{}
}

// Init initializes the tan operator.
func (t *Tan7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the tan operator.
func (t *Tan7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (t *Tan7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(t, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (t *Tan7) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (t *Tan7) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (t *Tan7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (t *Tan7) String() string {
	return "tan7 operator"
}

func tan[T ops.FloatType](x T) T {
	return T(math.Tan(float64(x)))
}
