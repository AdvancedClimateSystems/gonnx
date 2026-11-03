package ops

import (
	"math"
	"slices"

	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Reduction identifies how the elements along reduced axes are combined.
type Reduction int

// The supported reductions.
const (
	ReduceSum Reduction = iota
	ReduceMean
	ReduceMax
	ReduceMin
)

// Reduce combines the elements of t along the given axes. Without axes all axes are
// reduced. With keepDims the reduced axes remain as dimensions of size 1.
func Reduce(t *tensor.Tensor, reduction Reduction, axes []int, keepDims bool) (*tensor.Tensor, error) {
	switch data := t.Data().(type) {
	case []float32:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []float64:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []int8:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []int16:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []int32:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []int64:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []uint8:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []uint16:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []uint32:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	case []uint64:
		return reduce(data, t.Shape(), reduction, axes, keepDims)
	default:
		return nil, ErrInvalidType(t.DataType())
	}
}

func reduce[T tensor.Numeric](data []T, shape tensor.Shape, reduction Reduction, axes []int, keepDims bool) (*tensor.Tensor, error) {
	reduced, err := reducedAxes(axes, len(shape))
	if err != nil {
		return nil, err
	}

	keptShape := shape.Clone()
	count := 1

	for axis, isReduced := range reduced {
		if isReduced {
			count *= shape[axis]
			keptShape[axis] = 1
		}
	}

	// Every element of the input contributes to the output element at the same index
	// with the reduced coordinates set to 0. Walking the input in row-major order and
	// reading the output with stride 0 along reduced axes visits exactly those pairs.
	outStrides := contiguousStrides(keptShape)

	for axis, isReduced := range reduced {
		if isReduced {
			outStrides[axis] = 0
		}
	}

	out := make([]T, keptShape.Size())

	switch reduction {
	case ReduceSum, ReduceMean:
		accumulate(out, outStrides, data, shape, func(a, b T) T { return a + b })
	case ReduceMax:
		stridedCopy(out, data, keptShape, contiguousStrides(shape), 0)
		accumulate(out, outStrides, data, shape, func(a, b T) T { return max(a, b) })
	case ReduceMin:
		stridedCopy(out, data, keptShape, contiguousStrides(shape), 0)
		accumulate(out, outStrides, data, shape, func(a, b T) T { return min(a, b) })
	}

	if reduction == ReduceMean {
		for i := range out {
			out[i] /= T(count)
		}
	}

	if keepDims {
		return tensor.New(out, keptShape...), nil
	}

	outShape := make([]int, 0, len(shape))

	for axis, size := range shape {
		if !reduced[axis] {
			outShape = append(outShape, size)
		}
	}

	return tensor.New(out, outShape...), nil
}

// reducedAxes returns for every axis of a tensor with the given rank whether it is in axes.
// No axes means all axes.
func reducedAxes(axes []int, rank int) ([]bool, error) {
	reduced := make([]bool, rank)

	for _, axis := range axes {
		positive := ConvertNegativeAxis(axis, rank)
		if positive < 0 || positive >= rank {
			return nil, ErrAxisOutOfRange(-rank, rank, axis)
		}

		reduced[positive] = true
	}

	if len(axes) == 0 {
		for i := range reduced {
			reduced[i] = true
		}
	}

	return reduced, nil
}

// accumulate walks in, a row-major array with the given shape, and combines every element
// into out[offset], where the offset is computed with outStrides.
func accumulate[T any](out []T, outStrides []int, in []T, shape []int, combine func(a, b T) T) {
	if len(in) == 0 {
		return
	}

	shape, strides := coalesce(shape, outStrides)
	outStrides = strides[0]

	rank := len(shape)
	if rank == 0 {
		out[0] = combine(out[0], in[0])

		return
	}

	inner := shape[rank-1]
	innerStride := outStrides[rank-1]
	index := make([]int, rank-1)
	outOffset := 0

	for start := 0; start < len(in); start += inner {
		row := in[start : start+inner]
		position := outOffset

		for _, value := range row {
			out[position] = combine(out[position], value)
			position += innerStride
		}

		for axis := rank - 2; axis >= 0; axis-- {
			index[axis]++
			outOffset += outStrides[axis]

			if index[axis] < shape[axis] {
				break
			}

			outOffset -= outStrides[axis] * shape[axis]
			index[axis] = 0
		}
	}
}

// alongAxis splits a shape into the number of elements before an axis, the size of the
// axis and the number of elements after it.
func alongAxis(shape tensor.Shape, axis int) (int, int, int) {
	return shape[:axis].Size(), shape[axis], shape[axis+1:].Size()
}

// ArgMax returns the int64 index of the first maximum along an axis.
func ArgMax(t *tensor.Tensor, axis int, keepDims bool) (*tensor.Tensor, error) {
	rank := t.Rank()

	axis = ConvertNegativeAxis(axis, rank)
	if axis < 0 || axis >= rank {
		return nil, ErrAxisOutOfRange(-rank, rank, axis)
	}

	var indices []int64

	switch data := t.Data().(type) {
	case []float32:
		indices = argMax(data, t.Shape(), axis)
	case []float64:
		indices = argMax(data, t.Shape(), axis)
	case []int8:
		indices = argMax(data, t.Shape(), axis)
	case []int16:
		indices = argMax(data, t.Shape(), axis)
	case []int32:
		indices = argMax(data, t.Shape(), axis)
	case []int64:
		indices = argMax(data, t.Shape(), axis)
	case []uint8:
		indices = argMax(data, t.Shape(), axis)
	case []uint16:
		indices = argMax(data, t.Shape(), axis)
	case []uint32:
		indices = argMax(data, t.Shape(), axis)
	case []uint64:
		indices = argMax(data, t.Shape(), axis)
	default:
		return nil, ErrInvalidType(t.DataType())
	}

	shape := t.Shape().Clone()
	if keepDims {
		shape[axis] = 1
	} else {
		shape = slices.Delete(shape, axis, axis+1)
	}

	return tensor.New(indices, shape...), nil
}

func argMax[T tensor.Numeric](data []T, shape tensor.Shape, axis int) []int64 {
	outer, axisSize, inner := alongAxis(shape, axis)
	out := make([]int64, outer*inner)

	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			base := o*axisSize*inner + i
			best := data[base]
			bestIndex := 0

			for a := 1; a < axisSize; a++ {
				if value := data[base+a*inner]; value > best {
					best = value
					bestIndex = a
				}
			}

			out[o*inner+i] = int64(bestIndex)
		}
	}

	return out
}

// Softmax computes exp(x) / sum(exp(x)) along an axis. With log set, it computes the
// logarithm of that instead.
func Softmax(t *tensor.Tensor, axis int, log bool) (*tensor.Tensor, error) {
	switch data := t.Data().(type) {
	case []float32:
		return tensor.New(softmax(data, t.Shape(), axis, log), t.Shape()...), nil
	case []float64:
		return tensor.New(softmax(data, t.Shape(), axis, log), t.Shape()...), nil
	default:
		return nil, ErrInvalidType(t.DataType())
	}
}

func softmax[T FloatType](data []T, shape tensor.Shape, axis int, log bool) []T {
	outer, axisSize, inner := alongAxis(shape, axis)
	out := make([]T, len(data))

	// The exponentials are kept in float64 so every output is rounded only once.
	exps := make([]float64, axisSize)

	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			base := o*axisSize*inner + i

			maximum := data[base]
			for a := 1; a < axisSize; a++ {
				maximum = max(maximum, data[base+a*inner])
			}

			var sum float64

			for a := range exps {
				exps[a] = math.Exp(float64(data[base+a*inner]) - float64(maximum))
				sum += exps[a]
			}

			if log {
				logSum := math.Log(sum)
				for a := range exps {
					out[base+a*inner] = T(float64(data[base+a*inner]) - float64(maximum) - logSum)
				}

				continue
			}

			for a, exp := range exps {
				out[base+a*inner] = T(exp / sum)
			}
		}
	}

	return out
}

// CumSum computes the cumulative sum along an axis. With exclusive set, an element does
// not include itself in its sum; with reverse set, the sum runs from the end of the axis.
func CumSum(t *tensor.Tensor, axis int, exclusive, reverse bool) (*tensor.Tensor, error) {
	rank := t.Rank()

	positive := ConvertNegativeAxis(axis, rank)
	if positive < 0 || positive >= rank {
		return nil, ErrAxisOutOfRange(0, rank, axis)
	}

	switch data := t.Data().(type) {
	case []float32:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	case []float64:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	case []int32:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	case []int64:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	case []uint32:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	case []uint64:
		return tensor.New(cumSum(data, t.Shape(), positive, exclusive, reverse), t.Shape()...), nil
	default:
		return nil, ErrInvalidType(t.DataType())
	}
}

func cumSum[T tensor.Numeric](data []T, shape tensor.Shape, axis int, exclusive, reverse bool) []T {
	outer, axisSize, inner := alongAxis(shape, axis)
	out := make([]T, len(data))

	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			base := o*axisSize*inner + i

			var sum T

			for step := 0; step < axisSize; step++ {
				a := step
				if reverse {
					a = axisSize - 1 - step
				}

				position := base + a*inner

				if exclusive {
					out[position] = sum
					sum += data[position]
				} else {
					sum += data[position]
					out[position] = sum
				}
			}
		}
	}

	return out
}
