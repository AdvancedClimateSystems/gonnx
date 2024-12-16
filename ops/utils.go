package ops

import (
	"gorgonia.org/tensor"
)

// Abs returns the absolute value of an int.
func Abs(x int) int {
	if x < 0 {
		x *= -1
	}

	return x
}

// Int64ToBool converts a int64 to a boolean.
func Int64ToBool(v int64) bool {
	return !(v == 0)
}

// AllInRange checks if all the entries in `arr` are in the inclusive range min <= x <= max.
func AllInRange(arr []int, minVal, maxVal int) bool {
	for _, ax := range arr {
		if ax < minVal || ax > maxVal {
			return false
		}
	}

	return true
}

// HasDuplicates checks if there are duplicates in the sorted array `arr`.
func HasDuplicates(arr []int) bool {
	if len(arr) < 1 {
		return false
	}

	prev := arr[0]

	for _, x := range arr[1:] {
		if prev == x {
			return true
		}

		prev = x
	}

	return false
}

// OffsetArrayIfNegative adds `offset` to negative elements in the array `arr`.
// `arr` is modified in place.
func OffsetArrayIfNegative(arr []int, offset int) {
	for i, ax := range arr {
		if ax < 0 {
			ax += offset
		}

		arr[i] = ax
	}
}

// OffsetTensorIfNegative adds an offset to every negative element in tensor t.
// Works only for tensors with Dtype int (same as offset).
func OffsetTensorIfNegative(t tensor.Tensor, offset int) error {
	f := func(n int) int {
		if n < 0 {
			return n + offset
		}

		return n
	}

	if _, err := t.Apply(f, tensor.WithReuse(t)); err != nil {
		return err
	}

	return nil
}

// AnyToIntSlice casts the data of a node to an int list. This will only
// be done if the data is of some sort of int type.
func AnyToIntSlice(value interface{}) ([]int, error) {
	var res []int

	switch data := value.(type) {
	case []int8:
		for _, value := range data {
			res = append(res, int(value))
		}

		return res, nil
	case []int16:
		for _, value := range data {
			res = append(res, int(value))
		}

		return res, nil
	case []int32:
		for _, value := range data {
			res = append(res, int(value))
		}

		return res, nil
	case []int64:
		for _, value := range data {
			res = append(res, int(value))
		}

		return res, nil
	default:
		return nil, ErrCast
	}
}

// GetValueAsTensorType returns the given value as the given tensor type.
func GetValueAsTensorType(value float64, dtype tensor.Dtype) (interface{}, error) {
	switch dtype {
	case tensor.Bool:
		return value > 0.0, nil
	case tensor.Int8:
		return int8(value), nil
	case tensor.Int16:
		return int16(value), nil
	case tensor.Int32:
		return int32(value), nil
	case tensor.Int64:
		return int64(value), nil
	case tensor.Float32:
		return float32(value), nil
	case tensor.Float64:
		return value, nil
	default:
		return nil, ErrCast
	}
}

// IfScalarToSlice will wrap the value in a slice if it is a scalar in a slice with that value,
// otherwise will return itself.
func IfScalarToSlice(value any) any {
	switch data := value.(type) {
	case int8:
		return []int8{data}
	case int16:
		return []int16{data}
	case int32:
		return []int32{data}
	case int64:
		return []int64{data}
	case int:
		return []int{data}
	case float32:
		return []float32{data}
	case float64:
		return []float64{data}
	case complex64:
		return []complex64{data}
	case complex128:
		return []complex128{data}
	default:
		return value
	}
}

// Zeros fills a float32 slice with 0's.
func Zeros(size int) []float32 {
	res := make([]float32, size)
	for i := range res {
		res[i] = 0.0
	}

	return res
}

// Full fills a slice with value, named after np.full.
func Full(size int, value float32) []float32 {
	res := make([]float32, size)
	for i := range res {
		res[i] = value
	}

	return res
}

// Ones fills a slice with float32 ones.
func Ones(size int) []float32 {
	res := make([]float32, size)
	for i := range res {
		res[i] = 1.0
	}

	return res
}

// Arange fills a slice with float32 ranging from 0, to size with step, step.
func Arange(size int, step float32) []float32 {
	res := make([]float32, size)
	for i := range res {
		res[i] = float32(i) * step
	}

	return res
}

// NElements calculates the amount of elements in a tensor based on its shape.
func NElements(shp ...int) int {
	nElem := 1
	for _, s := range shp {
		nElem *= s
	}

	return nElem
}

// PairwiseAssign essentially does pairwise t1 = t2 in place!.
func PairwiseAssign(t1, t2 tensor.Tensor) (err error) {
	if !t1.Shape().Eq(t2.Shape()) {
		return ErrInvalidShape
	}

	it := t1.Iterator()
	// We cannot check the error here since it is a post statement so ignore the nolint errcheck here.
	// nolint errcheck
	for it.Reset(); !it.Done(); it.Next() {
		coord := it.Coord()

		value, err := t2.At(coord...)
		if err != nil {
			return err
		}

		err = t1.SetAt(value, coord...)
		if err != nil {
			return err
		}
	}

	return nil
}

// Converts a negative axis to the corresponding axis such that it can be used as index.
// For example, if axis is -1, this represents the last dimension. Go does not support
// negative indexing (as opposed to Python, on which ONNX is heavily dependent), so we
// have to convert the negative axis to the positive axis it represents, which is dependent
// on the rank (number of dimensions) of the tensor.
// Example 1: if rank is 3, and axis is -1, the corresponding positive axis is 2.
// Example 2: if rank is 4, and axis is -1, the corresponding positive axis is 3.
// Example 3: if rank is 4, and axis is -3, the corresponding positive axis is 1.
// Example 4: if rank is 3, and axis is 2, the function does nothing.
func ConvertNegativeAxis(axis, rank int) int {
	if axis < 0 {
		axis = rank + axis
	}

	return axis
}
