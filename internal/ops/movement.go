package ops

import (
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Strided returns a tensor of the given shape whose element at index (i0, i1, ...) is the
// element of t at offset + i0*strides[0] + i1*strides[1] + ... in row-major order.
func Strided(t *tensor.Tensor, shape, strides []int, offset int) (*tensor.Tensor, error) {
	switch data := t.Data().(type) {
	case []float32:
		return strided(data, shape, strides, offset), nil
	case []float64:
		return strided(data, shape, strides, offset), nil
	case []int8:
		return strided(data, shape, strides, offset), nil
	case []int16:
		return strided(data, shape, strides, offset), nil
	case []int32:
		return strided(data, shape, strides, offset), nil
	case []int64:
		return strided(data, shape, strides, offset), nil
	case []uint8:
		return strided(data, shape, strides, offset), nil
	case []uint16:
		return strided(data, shape, strides, offset), nil
	case []uint32:
		return strided(data, shape, strides, offset), nil
	case []uint64:
		return strided(data, shape, strides, offset), nil
	case []bool:
		return strided(data, shape, strides, offset), nil
	default:
		return nil, ErrInvalidType(t.DataType())
	}
}

func strided[T tensor.Element](data []T, shape, strides []int, offset int) *tensor.Tensor {
	out := make([]T, tensor.Shape(shape).Size())
	stridedCopy(out, data, shape, strides, offset)

	return tensor.New(out, shape...)
}

// Transpose permutes the dimensions of t: dimension i of the result is dimension perm[i]
// of t. Without a permutation the order of the dimensions is reversed.
func Transpose(t *tensor.Tensor, perm ...int) (*tensor.Tensor, error) {
	rank := t.Rank()

	if len(perm) == 0 {
		perm = make([]int, rank)
		for i := range perm {
			perm[i] = rank - 1 - i
		}
	}

	if !isPermutation(perm, rank) {
		return nil, ErrInvalidPermutation(perm, rank)
	}

	inStrides := contiguousStrides(t.Shape())
	shape := make([]int, rank)
	strides := make([]int, rank)

	for i, axis := range perm {
		shape[i] = t.Shape()[axis]
		strides[i] = inStrides[axis]
	}

	return Strided(t, shape, strides, 0)
}

func isPermutation(perm []int, rank int) bool {
	if len(perm) != rank {
		return false
	}

	seen := make([]bool, rank)

	for _, axis := range perm {
		if axis < 0 || axis >= rank || seen[axis] {
			return false
		}

		seen[axis] = true
	}

	return true
}

// Slice returns the elements of t selected per axis by start, end and step, following
// the ONNX Slice semantics: negative starts and ends count from the end of the axis, and
// out of range values are clamped.
func Slice(t *tensor.Tensor, starts, ends, axes, steps []int) (*tensor.Tensor, error) {
	rank := t.Rank()
	shape := t.Shape().Clone()
	inStrides := contiguousStrides(t.Shape())
	strides := append([]int(nil), inStrides...)
	offset := 0

	for i, axis := range axes {
		axis = ConvertNegativeAxis(axis, rank)
		if axis < 0 || axis >= rank {
			return nil, ErrAxisOutOfRange(-rank, rank, axes[i])
		}

		step := steps[i]
		if step == 0 {
			return nil, ErrDimension("slice step cannot be 0")
		}

		start, count := sliceRange(starts[i], ends[i], step, t.Shape()[axis])
		shape[axis] = count
		offset += start * inStrides[axis]
		strides[axis] = inStrides[axis] * step
	}

	return Strided(t, shape, strides, offset)
}

// sliceRange returns the first index and number of elements that slicing an axis of the
// given size from start to end with step selects.
func sliceRange(start, end, step, size int) (int, int) {
	if start < 0 {
		start += size
	}

	if end < 0 {
		end += size
	}

	if step > 0 {
		start = clamp(start, 0, size)
		end = clamp(end, 0, size)

		if end <= start {
			return start, 0
		}

		return start, (end - start + step - 1) / step
	}

	start = clamp(start, 0, size-1)
	end = clamp(end, -1, size-1)

	if start <= end {
		return start, 0
	}

	return start, (start - end - step - 1) / -step
}

func clamp(value, low, high int) int {
	return max(low, min(value, high))
}

// Concat joins tensors along an existing axis. All tensors must have the same data type,
// and the same shape except for the given axis.
func Concat(axis int, tensors ...*tensor.Tensor) (*tensor.Tensor, error) {
	first := tensors[0]
	rank := first.Rank()

	axis = ConvertNegativeAxis(axis, rank)
	if axis < 0 || axis >= rank {
		return nil, ErrAxisOutOfRange(-rank, rank, axis)
	}

	shape := first.Shape().Clone()
	shape[axis] = 0

	for _, t := range tensors {
		if t.DataType() != first.DataType() {
			return nil, ErrDataTypeMismatch(first.DataType(), t.DataType())
		}

		if t.Rank() != rank {
			return nil, ErrIncompatibleDimensions()
		}

		for i, size := range t.Shape() {
			if i != axis && size != first.Shape()[i] {
				return nil, ErrIncompatibleDimensions()
			}
		}

		shape[axis] += t.Shape()[axis]
	}

	switch first.DataType() {
	case tensor.Float32:
		return concat[float32](tensors, shape, axis), nil
	case tensor.Float64:
		return concat[float64](tensors, shape, axis), nil
	case tensor.Int8:
		return concat[int8](tensors, shape, axis), nil
	case tensor.Int16:
		return concat[int16](tensors, shape, axis), nil
	case tensor.Int32:
		return concat[int32](tensors, shape, axis), nil
	case tensor.Int64:
		return concat[int64](tensors, shape, axis), nil
	case tensor.Uint8:
		return concat[uint8](tensors, shape, axis), nil
	case tensor.Uint16:
		return concat[uint16](tensors, shape, axis), nil
	case tensor.Uint32:
		return concat[uint32](tensors, shape, axis), nil
	case tensor.Uint64:
		return concat[uint64](tensors, shape, axis), nil
	case tensor.Bool:
		return concat[bool](tensors, shape, axis), nil
	default:
		return nil, ErrInvalidType(first.DataType())
	}
}

func concat[T tensor.Element](tensors []*tensor.Tensor, shape tensor.Shape, axis int) *tensor.Tensor {
	out := make([]T, shape.Size())
	outer := shape[:axis].Size()
	position := 0

	for i := 0; i < outer; i++ {
		for _, t := range tensors {
			chunk := t.Shape()[axis:].Size()
			position += copy(out[position:], tensor.Data[T](t)[i*chunk:(i+1)*chunk])
		}
	}

	return tensor.New(out, shape...)
}

// Gather selects slices of data along axis: the result has the shape
// data.shape[:axis] + indicesShape + data.shape[axis+1:]. The indices must be within the
// size of the axis and non-negative.
func Gather(data *tensor.Tensor, indices []int, indicesShape tensor.Shape, axis int) (*tensor.Tensor, error) {
	switch d := data.Data().(type) {
	case []float32:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []float64:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []int8:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []int16:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []int32:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []int64:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []uint8:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []uint16:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []uint32:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []uint64:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	case []bool:
		return gather(d, data.Shape(), indices, indicesShape, axis), nil
	default:
		return nil, ErrInvalidType(data.DataType())
	}
}

func gather[T tensor.Element](data []T, shape tensor.Shape, indices []int, indicesShape tensor.Shape, axis int) *tensor.Tensor {
	outShape := append(append(shape[:axis].Clone(), indicesShape...), shape[axis+1:]...)
	out := make([]T, outShape.Size())

	outer := shape[:axis].Size()
	inner := shape[axis+1:].Size()
	axisSize := shape[axis]
	position := 0

	for i := 0; i < outer; i++ {
		for _, index := range indices {
			start := (i*axisSize + index) * inner
			position += copy(out[position:], data[start:start+inner])
		}
	}

	return tensor.New(out, outShape...)
}
