package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinAndInputs = 2
	MaxAndInputs = 2
)

// And represents the ONNX and operator.
type And struct{}

// newAnd creates a new and operator.
func newAnd() ops.Operator {
	return &And{}
}

// Init initializes the and operator.
func (a *And) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the and operator.
func (a *And) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := and(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (a *And) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(a, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (a *And) GetMinInputs() int {
	return MinAndInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (a *And) GetMaxInputs() int {
	return MaxAndInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (a *And) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (a *And) String() string {
	return "and operator"
}

func and(A, B tensor.Tensor) (tensor.Tensor, error) {
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

		output.SetAt(boolA && boolB, iterator.Coord()...)
	}

	return output, nil
}
