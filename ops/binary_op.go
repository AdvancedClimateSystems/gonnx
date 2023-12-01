package ops

import (
	"gorgonia.org/tensor"
)

// BinaryOp describes a general operation between 2 tensors with 1 tensor as result.
type BinaryOp func(A, B tensor.Tensor) (tensor.Tensor, error)

// ApplyBinaryOperation applies a binary operation (an operation of arity 2) to 2 tensors.
// It returns a list of tensors with only 1 output tensor in order for this function to
// be easily used in operators.
func ApplyBinaryOperation(A, B tensor.Tensor, op BinaryOp, broadcastOption BroadcastType) ([]tensor.Tensor, error) {
	var err error

	switch broadcastOption {
	case NoBroadcasting:
		break
	case UnidirectionalBroadcasting:
		A, B, err = UnidirectionalBroadcast(A, B)
		if err != nil {
			return nil, err
		}
	case MultidirectionalBroadcasting:
		A, B, err = MultidirectionalBroadcast(A, B)
		if err != nil {
			return nil, err
		}
	}

	out, err := op(A, B)

	return []tensor.Tensor{out}, err
}

// Or applies the boolean 'or' operation on 2 tensors.
func Or(A, B tensor.Tensor) (tensor.Tensor, error) {
	return applyBooleanBinaryOperator(
		A,
		B,
		func(a, b bool) bool { return a || b },
	)
}

// And applies the boolean 'and' operation on 2 tensors.
func And(A, B tensor.Tensor) (tensor.Tensor, error) {
	return applyBooleanBinaryOperator(
		A,
		B,
		func(a, b bool) bool { return a && b },
	)
}

// Xor applies the boolean 'xor' operation on 2 tensors.
func Xor(A, B tensor.Tensor) (tensor.Tensor, error) {
	return applyBooleanBinaryOperator(
		A,
		B,
		func(a, b bool) bool { return a != b },
	)
}

// BooleanOp describes a binary operation between two booleans that also returns a boolean.
type BooleanOp func(a, b bool) bool

// ApplyBooleanOperator is a function that applies a boolean operator element-wise to
// to 2 tensors. This assumes that A and B have exactly the same shape.
// We create an iterator that loops over all elements of A (which can also be used for B).
// Using this iterator, the given boolean operator is applied to all pairs of elements from
// A and B and the result is returned.
func applyBooleanBinaryOperator(A, B tensor.Tensor, op BooleanOp) (tensor.Tensor, error) {
	A, B, err := MultidirectionalBroadcast(A, B)
	if err != nil {
		return nil, err
	}

	output := tensor.NewDense(tensor.Bool, A.Shape())
	output.Zero()

	iterator := A.Iterator()
	iterator.Reset()

	for !iterator.Done() {
		valA, err := A.At(iterator.Coord()...)
		if err != nil {
			return nil, err
		}

		boolA, ok := valA.(bool)
		if !ok {
			return nil, ErrTypeAssert("bool", valA)
		}

		valB, err := B.At(iterator.Coord()...)
		if err != nil {
			return nil, err
		}

		boolB, ok := valB.(bool)
		if !ok {
			return nil, ErrTypeAssert("bool", valB)
		}

		err = output.SetAt(op(boolA, boolB), iterator.Coord()...)
		if err != nil {
			return nil, err
		}

		_, err = iterator.Next()
		if err != nil {
			return nil, err
		}
	}

	return output, nil
}
