package ops

// contiguousStrides returns the element strides of a row-major array with the given shape.
func contiguousStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1

	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return strides
}

// coalesce merges neighbouring dimensions that every operand walks through as one
// contiguous run, which lengthens the innermost loop of the strided kernels. Size 1
// dimensions are dropped because they never advance an index.
func coalesce(shape []int, strides ...[]int) ([]int, [][]int) {
	newShape := make([]int, 0, len(shape))
	newStrides := make([][]int, len(strides))

	for i := range strides {
		newStrides[i] = make([]int, 0, len(shape))
	}

	for axis, size := range shape {
		if size == 1 {
			continue
		}

		last := len(newShape) - 1
		mergeable := last >= 0

		for i := range strides {
			if mergeable && newStrides[i][last] != strides[i][axis]*size {
				mergeable = false
			}
		}

		if mergeable {
			newShape[last] *= size

			for i := range strides {
				newStrides[i][last] = strides[i][axis]
			}

			continue
		}

		newShape = append(newShape, size)

		for i := range strides {
			newStrides[i] = append(newStrides[i], strides[i][axis])
		}
	}

	return newShape, newStrides
}

// stridedCopy fills dst, a row-major array with the given shape, with elements of src.
// The element at index (i0, i1, ...) is read from src[offset + i0*strides[0] + ...].
func stridedCopy[T any](dst, src []T, shape, strides []int, offset int) {
	if len(dst) == 0 {
		return
	}

	shape, coalesced := coalesce(shape, strides)
	strides = coalesced[0]

	rank := len(shape)
	if rank == 0 {
		dst[0] = src[offset]

		return
	}

	inner := shape[rank-1]
	innerStride := strides[rank-1]
	index := make([]int, rank-1)
	srcOffset := offset

	for start := 0; start < len(dst); start += inner {
		row := dst[start : start+inner]

		switch innerStride {
		case 1:
			copy(row, src[srcOffset:srcOffset+inner])
		case 0:
			value := src[srcOffset]
			for i := range row {
				row[i] = value
			}
		default:
			position := srcOffset
			for i := range row {
				row[i] = src[position]
				position += innerStride
			}
		}

		for axis := rank - 2; axis >= 0; axis-- {
			index[axis]++
			srcOffset += strides[axis]

			if index[axis] < shape[axis] {
				break
			}

			srcOffset -= strides[axis] * shape[axis]
			index[axis] = 0
		}
	}
}
