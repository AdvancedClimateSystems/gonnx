package prelu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	PRelu7MinInputs = 2
	PRelu7MaxInputs = 2
)

// PRelu7 represents the ONNX prelu operator.
type PRelu7 struct{}

// newPRelu7 creates a new prelu operator.
func newPRelu7() ops.Operator {
	return &PRelu7{}
}

// Init initializes the prelu operator.
func (op *PRelu7) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the prelu operator.
func (op *PRelu7) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var err error

	x, slope := inputs[0], inputs[1]

	x, slope, err = ops.UnidirectionalBroadcast(x, slope)
	if err != nil {
		return nil, err
	}

	y := tensor.NewDense(x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.Float32:
		err = calcPRelu[float32](y.Data(), x.Data(), slope.Data())
	case tensor.Float64:
		err = calcPRelu[float64](y.Data(), x.Data(), slope.Data())
	case tensor.Uint32:
		err = calcPRelu[uint32](y.Data(), x.Data(), slope.Data())
	case tensor.Uint64:
		err = calcPRelu[uint64](y.Data(), x.Data(), slope.Data())
	case tensor.Int32:
		err = calcPRelu[int32](y.Data(), x.Data(), slope.Data())
	case tensor.Int64:
		err = calcPRelu[int64](y.Data(), x.Data(), slope.Data())
	default:
		return nil, ops.ErrInvalidInputType(0, x.Dtype().String(), op)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{y}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (op *PRelu7) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	inputs, err := ops.ValidateInputs(op, inputs)
	if err != nil {
		return nil, err
	}

	x, slope := inputs[0], inputs[1]
	if x.Dtype() != slope.Dtype() {
		return nil, ops.ErrInvalidTensor("DType of 'slope' does not match DType of 'x'", op)
	}

	return inputs, nil
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (op *PRelu7) GetMinInputs() int {
	return PRelu7MinInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (op *PRelu7) GetMaxInputs() int {
	return PRelu7MaxInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (op *PRelu7) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (op *PRelu7) String() string {
	return "prelu7 operator"
}
