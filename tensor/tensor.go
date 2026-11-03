// Package tensor provides the n-dimensional array that gonnx models consume and produce.
//
// A Tensor is a dense, row-major array of a single element type. Reshaping a tensor is
// cheap: the result shares its elements with the original. Code that writes into the
// slice returned by Data should therefore only do so for tensors it created itself.
package tensor

import (
	"errors"
	"fmt"
)

// ErrShapeMismatch is returned when a shape does not fit the number of elements of a tensor.
var ErrShapeMismatch = errors.New("tensor: shape does not match number of elements")

// Tensor is a dense n-dimensional array. A tensor with an empty shape is a scalar and
// holds exactly one element.
type Tensor struct {
	shape    Shape
	dataType DataType
	data     any
}

// New creates a tensor that takes ownership of data, interpreted in row-major order with
// the given shape. Omitting the shape creates a scalar. New panics when the number of
// elements in data does not match the shape.
func New[T Element](data []T, shape ...int) *Tensor {
	s := Shape(shape).Clone()
	if len(data) != s.Size() {
		panic(fmt.Errorf("%w: %d elements, shape %v", ErrShapeMismatch, len(data), s))
	}

	return &Tensor{shape: s, dataType: dataTypeOf[T](), data: data}
}

// Scalar creates a tensor with an empty shape holding a single value.
func Scalar[T Element](value T) *Tensor {
	return New([]T{value})
}

// Zeros creates a tensor of the given data type and shape with all elements set to their
// zero value. Zeros panics for data types this package does not support.
func Zeros(dataType DataType, shape ...int) *Tensor {
	size := Shape(shape).Size()

	switch dataType {
	case Float32:
		return New(make([]float32, size), shape...)
	case Float64:
		return New(make([]float64, size), shape...)
	case Int8:
		return New(make([]int8, size), shape...)
	case Int16:
		return New(make([]int16, size), shape...)
	case Int32:
		return New(make([]int32, size), shape...)
	case Int64:
		return New(make([]int64, size), shape...)
	case Uint8:
		return New(make([]uint8, size), shape...)
	case Uint16:
		return New(make([]uint16, size), shape...)
	case Uint32:
		return New(make([]uint32, size), shape...)
	case Uint64:
		return New(make([]uint64, size), shape...)
	case Bool:
		return New(make([]bool, size), shape...)
	default:
		panic(fmt.Sprintf("tensor: unsupported data type %v", dataType))
	}
}

// Data returns the elements of t as a []T. It panics when T does not match the data type
// of t; use the Data method with a type assertion or switch when the type is not known.
func Data[T Element](t *Tensor) []T {
	data, ok := t.data.([]T)
	if !ok {
		panic(fmt.Sprintf("tensor: cannot access %v tensor as %T", t.dataType, data))
	}

	return data
}

// Data returns the elements of the tensor as a typed slice ([]float32, []int64, ...).
func (t *Tensor) Data() any {
	return t.data
}

// Shape returns the shape of the tensor. The returned slice must not be modified.
func (t *Tensor) Shape() Shape {
	return t.shape
}

// Rank returns the number of dimensions of the tensor.
func (t *Tensor) Rank() int {
	return len(t.shape)
}

// Size returns the number of elements in the tensor.
func (t *Tensor) Size() int {
	return t.shape.Size()
}

// DataType returns the element type of the tensor.
func (t *Tensor) DataType() DataType {
	return t.dataType
}

// Reshape returns a tensor with the given shape that shares its elements with t.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	s := Shape(shape).Clone()
	if s.Size() != t.Size() {
		return nil, fmt.Errorf("%w: cannot reshape %v into %v", ErrShapeMismatch, t.shape, s)
	}

	return &Tensor{shape: s, dataType: t.dataType, data: t.data}, nil
}

// Clone returns a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	var data any

	switch d := t.data.(type) {
	case []float32:
		data = append([]float32(nil), d...)
	case []float64:
		data = append([]float64(nil), d...)
	case []int8:
		data = append([]int8(nil), d...)
	case []int16:
		data = append([]int16(nil), d...)
	case []int32:
		data = append([]int32(nil), d...)
	case []int64:
		data = append([]int64(nil), d...)
	case []uint8:
		data = append([]uint8(nil), d...)
	case []uint16:
		data = append([]uint16(nil), d...)
	case []uint32:
		data = append([]uint32(nil), d...)
	case []uint64:
		data = append([]uint64(nil), d...)
	case []bool:
		data = append([]bool(nil), d...)
	}

	return &Tensor{shape: t.shape.Clone(), dataType: t.dataType, data: data}
}

// String returns a short description of the tensor: its data type, shape and elements.
func (t *Tensor) String() string {
	return fmt.Sprintf("tensor(%v, %v, %v)", t.dataType, []int(t.shape), t.data)
}
