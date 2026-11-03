package ops

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/tensor"
)

// BinaryOp describes a general operation between 2 tensors with 1 tensor as result. The
// operation broadcasts its inputs multidirectionally.
type BinaryOp func(A, B *tensor.Tensor) (*tensor.Tensor, error)

// ApplyBinaryOperation applies a binary operation (an operation of arity 2) to 2 tensors,
// after checking that their shapes satisfy the given broadcast rule. It returns a list of
// tensors with only 1 output tensor in order for this function to be easily used in
// operators.
func ApplyBinaryOperation(A, B *tensor.Tensor, op BinaryOp, broadcastOption BroadcastType) ([]*tensor.Tensor, error) {
	switch broadcastOption {
	case NoBroadcasting:
		if !A.Shape().Equal(B.Shape()) {
			return nil, ErrIncompatibleDimensions()
		}
	case UnidirectionalBroadcasting:
		if _, err := UnidirectionalBroadcastShape(A.Shape(), B.Shape()); err != nil {
			return nil, err
		}
	case MultidirectionalBroadcasting:
		if _, err := MultidirectionalBroadcastShape(A.Shape(), B.Shape()); err != nil {
			return nil, err
		}
	}

	out, err := op(A, B)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}

type arithmetic int

const (
	addition arithmetic = iota
	subtraction
	multiplication
	division
	power
)

type comparison int

const (
	greater comparison = iota
	greaterOrEqual
	less
	lessOrEqual
	equal
)

// Add adds 2 tensors to each other.
func Add(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyArithmetic(A, B, addition)
}

// Sub subtracts 1 tensor from the other.
func Sub(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyArithmetic(A, B, subtraction)
}

// Mul multiplies 2 tensors with each other.
func Mul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyArithmetic(A, B, multiplication)
}

// Div divides 1 tensor by the other. Integer division truncates towards zero.
func Div(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyArithmetic(A, B, division)
}

// Pow raises the first tensor to the power of the second tensor. The power is computed in
// float64 and converted back to the element type of the tensors.
func Pow(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyArithmetic(A, B, power)
}

// Gt applies the greater than (>) operator on 2 tensors.
func Gt(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyComparison(A, B, greater)
}

// Gte applies the greater or equal than (>=) operator on 2 tensors.
func Gte(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyComparison(A, B, greaterOrEqual)
}

// Lt applies the less than (<) operator on 2 tensors.
func Lt(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyComparison(A, B, less)
}

// Lte applies the less or equal than (<=) operator on 2 tensors.
func Lte(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyComparison(A, B, lessOrEqual)
}

// Equal applies the equal operator (=) operator on 2 tensors.
func Equal(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if A.DataType() == tensor.Bool && B.DataType() == tensor.Bool {
		return applyBoolean(A, B, func(a, b bool) bool { return a == b })
	}

	return applyComparison(A, B, equal)
}

// Or applies the boolean 'or' operation on 2 tensors.
func Or(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyBoolean(A, B, func(a, b bool) bool { return a || b })
}

// And applies the boolean 'and' operation on 2 tensors.
func And(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyBoolean(A, B, func(a, b bool) bool { return a && b })
}

// Xor applies the boolean 'xor' operation on 2 tensors.
func Xor(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return applyBoolean(A, B, func(a, b bool) bool { return a != b })
}

func applyArithmetic(A, B *tensor.Tensor, op arithmetic) (*tensor.Tensor, error) {
	if A.DataType() != B.DataType() {
		return nil, ErrDataTypeMismatch(A.DataType(), B.DataType())
	}

	switch A.DataType() {
	case tensor.Float32:
		return Elementwise(A, B, arithmeticFunc[float32](op))
	case tensor.Float64:
		return Elementwise(A, B, arithmeticFunc[float64](op))
	case tensor.Int8:
		return Elementwise(A, B, arithmeticFunc[int8](op))
	case tensor.Int16:
		return Elementwise(A, B, arithmeticFunc[int16](op))
	case tensor.Int32:
		return Elementwise(A, B, arithmeticFunc[int32](op))
	case tensor.Int64:
		return Elementwise(A, B, arithmeticFunc[int64](op))
	case tensor.Uint8:
		return Elementwise(A, B, arithmeticFunc[uint8](op))
	case tensor.Uint16:
		return Elementwise(A, B, arithmeticFunc[uint16](op))
	case tensor.Uint32:
		return Elementwise(A, B, arithmeticFunc[uint32](op))
	case tensor.Uint64:
		return Elementwise(A, B, arithmeticFunc[uint64](op))
	default:
		return nil, ErrInvalidType(A.DataType())
	}
}

func arithmeticFunc[T tensor.Numeric](op arithmetic) func(a, b T) T {
	switch op {
	case addition:
		return func(a, b T) T { return a + b }
	case subtraction:
		return func(a, b T) T { return a - b }
	case multiplication:
		return func(a, b T) T { return a * b }
	case division:
		return func(a, b T) T { return a / b }
	default:
		return func(a, b T) T { return T(math.Pow(float64(a), float64(b))) }
	}
}

func applyComparison(A, B *tensor.Tensor, op comparison) (*tensor.Tensor, error) {
	if A.DataType() != B.DataType() {
		return nil, ErrDataTypeMismatch(A.DataType(), B.DataType())
	}

	switch A.DataType() {
	case tensor.Float32:
		return Elementwise(A, B, comparisonFunc[float32](op))
	case tensor.Float64:
		return Elementwise(A, B, comparisonFunc[float64](op))
	case tensor.Int8:
		return Elementwise(A, B, comparisonFunc[int8](op))
	case tensor.Int16:
		return Elementwise(A, B, comparisonFunc[int16](op))
	case tensor.Int32:
		return Elementwise(A, B, comparisonFunc[int32](op))
	case tensor.Int64:
		return Elementwise(A, B, comparisonFunc[int64](op))
	case tensor.Uint8:
		return Elementwise(A, B, comparisonFunc[uint8](op))
	case tensor.Uint16:
		return Elementwise(A, B, comparisonFunc[uint16](op))
	case tensor.Uint32:
		return Elementwise(A, B, comparisonFunc[uint32](op))
	case tensor.Uint64:
		return Elementwise(A, B, comparisonFunc[uint64](op))
	default:
		return nil, ErrInvalidType(A.DataType())
	}
}

func comparisonFunc[T tensor.Numeric](op comparison) func(a, b T) bool {
	switch op {
	case greater:
		return func(a, b T) bool { return a > b }
	case greaterOrEqual:
		return func(a, b T) bool { return a >= b }
	case less:
		return func(a, b T) bool { return a < b }
	case lessOrEqual:
		return func(a, b T) bool { return a <= b }
	default:
		return func(a, b T) bool { return a == b }
	}
}

func applyBoolean(A, B *tensor.Tensor, op func(a, b bool) bool) (*tensor.Tensor, error) {
	if A.DataType() != tensor.Bool || B.DataType() != tensor.Bool {
		return nil, ErrDataTypeMismatch(A.DataType(), B.DataType())
	}

	return Elementwise(A, B, op)
}

// Elementwise applies f to every pair of elements of A and B after broadcasting them
// multidirectionally. A and B must hold elements of type T.
func Elementwise[T, U tensor.Element](A, B *tensor.Tensor, f func(a, b T) U) (*tensor.Tensor, error) {
	shape, err := MultidirectionalBroadcastShape(A.Shape(), B.Shape())
	if err != nil {
		return nil, err
	}

	a := tensor.Data[T](A)
	b := tensor.Data[T](B)
	out := make([]U, shape.Size())

	switch {
	case len(a) == len(out) && len(b) == len(out):
		b = b[:len(a)]
		for i := range out {
			out[i] = f(a[i], b[i])
		}
	case len(b) == 1:
		value := b[0]
		a = a[:len(out)]

		for i := range out {
			out[i] = f(a[i], value)
		}
	case len(a) == 1:
		value := a[0]
		b = b[:len(out)]

		for i := range out {
			out[i] = f(value, b[i])
		}
	default:
		elementwiseBroadcast(out, shape, a, broadcastStrides(A.Shape(), shape), b, broadcastStrides(B.Shape(), shape), f)
	}

	return tensor.New(out, shape...), nil
}

// elementwiseBroadcast computes out[i] = f(a[...], b[...]) for every index of shape, where
// the elements of a and b are read with the given (broadcast) strides.
func elementwiseBroadcast[T, U any](out []U, shape []int, a []T, aStrides []int, b []T, bStrides []int, f func(a, b T) U) {
	if len(out) == 0 {
		return
	}

	shape, strides := coalesce(shape, aStrides, bStrides)
	aStrides, bStrides = strides[0], strides[1]

	rank := len(shape)
	inner := shape[rank-1]
	aStep, bStep := aStrides[rank-1], bStrides[rank-1]
	index := make([]int, rank-1)
	aOffset, bOffset := 0, 0

	for start := 0; start < len(out); start += inner {
		row := out[start : start+inner]
		aPosition, bPosition := aOffset, bOffset

		for i := range row {
			row[i] = f(a[aPosition], b[bPosition])
			aPosition += aStep
			bPosition += bStep
		}

		for axis := rank - 2; axis >= 0; axis-- {
			index[axis]++
			aOffset += aStrides[axis]
			bOffset += bStrides[axis]

			if index[axis] < shape[axis] {
				break
			}

			aOffset -= aStrides[axis] * shape[axis]
			bOffset -= bStrides[axis] * shape[axis]
			index[axis] = 0
		}
	}
}

// Map applies f to every element of t, which must hold elements of type T.
func Map[T, U tensor.Element](t *tensor.Tensor, f func(T) U) *tensor.Tensor {
	in := tensor.Data[T](t)
	out := make([]U, len(in))

	for i, value := range in {
		out[i] = f(value)
	}

	return tensor.New(out, t.Shape()...)
}
