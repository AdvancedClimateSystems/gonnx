package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	PReluMinInputs = 2
	PReluMaxInputs = 2
)

// PRelu represents the ONNX prelu operator.
type PRelu struct{}

// newPRelu creates a new prelu operator.
func newPRelu() ops.Operator {
	return &PRelu{}
}

// Init initializes the prelu operator.
func (op *PRelu) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the prelu operator.
func (op *PRelu) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var err error

	x, slope := inputs[0], inputs[1]

	x, slope, err = ops.UnidirectionalBroadcast(x, slope)
	if err != nil {
		return nil, err
	}

	y := tensor.NewDense(x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.Float32:
		calcPRelu(y.Data().([]float32), x.Data().([]float32), slope.Data().([]float32))
	case tensor.Float64:
		calcPRelu(y.Data().([]float64), x.Data().([]float64), slope.Data().([]float64))
	case tensor.Uint32:
		calcPRelu(y.Data().([]uint32), x.Data().([]uint32), slope.Data().([]uint32))
	case tensor.Uint64:
		calcPRelu(y.Data().([]uint64), x.Data().([]uint64), slope.Data().([]uint64))
	case tensor.Int32:
		calcPRelu(y.Data().([]int32), x.Data().([]int32), slope.Data().([]int32))
	case tensor.Int64:
		calcPRelu(y.Data().([]int64), x.Data().([]int64), slope.Data().([]int64))
	default:
		return nil, ops.ErrInvalidInputType(0, x.Dtype().String(), op)
	}

	return []tensor.Tensor{y}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (op *PRelu) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	inputs, err := ops.ValidateInputs(op, inputs)
	if err != nil {
		return nil, err
	}

	x, slope := inputs[0], inputs[1]
	if x.Dtype() != slope.Dtype() {
		return nil, fmt.Errorf("%v: type of slope (%s) does not match type of X (%s)", op, slope.Dtype(), x.Dtype())
	}

	return inputs, nil
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (op *PRelu) GetMinInputs() int {
	return PReluMinInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (op *PRelu) GetMaxInputs() int {
	return PReluMaxInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (op *PRelu) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (op *PRelu) String() string {
	return "prelu operator"
}

func calcPRelu[T float32 | float64 | uint32 | uint64 | int32 | int64](result []T, input []T, slope []T) {
	for i, v := range input {
		if v < 0 {
			v = slope[i] * v
		}

		result[i] = v
	}
}
