package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var (
	MinXorInputs = 2
	MaxXorInputs = 2
)

// Xor represents the ONNX xor operator.
type Xor struct{}

// newXor creates a new xor operator.
func newXor() ops.Operator {
	return &Xor{}
}

// Init initializes the xor operator.
func (x *Xor) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the xor operator.
func (x *Xor) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	in1, in2, err := ops.MultidirectionalBroadcast(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	out, err := xor(in1, in2)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (x *Xor) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(x, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (x *Xor) GetMinInputs() int {
	return MinXorInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (x *Xor) GetMaxInputs() int {
	return MaxXorInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (x *Xor) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (x *Xor) String() string {
	return "xor operator"
}

func xor(A, B tensor.Tensor) (tensor.Tensor, error) {
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

		output.SetAt(boolA != boolB, iterator.Coord()...)
	}

	return output, nil
}
