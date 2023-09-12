package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// ReduceMax represents the ONNX reduceMax operator.
type ReduceMax struct {
	axes     []int
	keepdims int
}

// newReduceMax creates a new reduceMax operator.
func newReduceMax() ops.Operator {
	return &ReduceMax{
		axes:     []int{},
		keepdims: 1,
	}
}

// Init initializes the reduceMax operator.
func (r *ReduceMax) Init(attributes []*onnx.AttributeProto) error {
	if len(attributes) == 0 || len(attributes) > 2 {
		return fmt.Errorf(ops.InvalidAttrCountErrTemplate, r, "1 or 2", len(attributes))
	}

	for _, attr := range attributes {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}
			r.axes = axes
		case "keepdims":
			r.keepdims = int(attr.GetI())
		default:
			return fmt.Errorf(ops.UnknownAttributeErrTemplate, r, attr.GetName())
		}
	}

	return nil
}

// Apply applies the reduceMax operator.
func (r *ReduceMax) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := inputs[0].Reduce(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *ReduceMax) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *ReduceMax) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *ReduceMax) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *ReduceMax) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint8, tensor.Int8, tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *ReduceMax) String() string {
	return "reduceMax operator"
}
