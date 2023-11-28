package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinOrInputs = 2
	MaxOrInputs = 2
)

// Or represents the ONNX or operator.
type Or struct{}

// newOr creates a new or operator.
func newOr() ops.Operator {
	return &Or{}
}

// Init initializes the or operator.
func (o *Or) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the or operator.
func (o *Or) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := or(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (o *Or) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(o, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (o *Or) GetMinInputs() int {
	return MinOrInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (o *Or) GetMaxInputs() int {
	return MaxOrInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (o *Or) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (o *Or) String() string {
	return "or operator"
}

func or(A, B tensor.Tensor) (tensor.Tensor, error) {
	output := tensor.NewDense(tensor.Bool, A.Shape())
	output.Zero()

	iterator := A.Iterator()
	iterator.Reset()

	for iterator.Reset(); !iterator.Done(); iterator.Next() {
		valA, err := A.At(iterator.Coord()...)
		if err != nil {
			return nil, err
		}

		boolA, ok := valA.(bool)
		if !ok {
			return nil, ops.ErrTypeAssert("bool", valA)
		}

		valB, err := B.At(iterator.Coord()...)
		if err != nil {
			return nil, err
		}

		boolB, ok := valB.(bool)
		if !ok {
			return nil, ops.ErrTypeAssert("bool", valB)
		}

		output.SetAt(boolA || boolB, iterator.Coord()...)
	}

	return output, nil
}
