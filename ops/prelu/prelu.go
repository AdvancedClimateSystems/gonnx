package prelu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var prelu7TypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var preluTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// PRelu represents the ONNX prelu operator.
type PRelu struct {
	ops.BaseOperator
}

// newPRelu creates a new prelu operator.
func newPRelu(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &PRelu{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"prelu",
		),
	}
}

// Init initializes the prelu operator.
func (op *PRelu) Init(*onnx.NodeProto) error {
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
		return nil, ops.ErrInvalidInputType(0, x.Dtype().String(), op.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{y}, nil
}

func calcPRelu[T float32 | float64 | uint32 | uint64 | int32 | int64](result any, input any, slope any) error {
	var convertedResult []T

	var convertedInput []T

	var convertedSlope []T

	convertedResult, ok := result.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", result)
	}

	convertedInput, ok = input.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", input)
	}

	convertedSlope, ok = slope.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", slope)
	}

	for i, v := range convertedInput {
		if v < 0 {
			v = convertedSlope[i] * v
		}

		convertedResult[i] = v
	}

	return nil
}
