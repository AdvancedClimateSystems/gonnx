package ops

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAbs(t *testing.T) {
	tests := []struct {
		in       int
		expected int
	}{{1, 1}, {-1, 1}, {-2, 2}}
	for _, test := range tests {
		assert.Equal(t, test.expected, Abs(test.in))
	}
}

func TestInt64ToBool(t *testing.T) {
	tests := []struct {
		in       int64
		expected bool
	}{
		{
			int64(1),
			true,
		},
		{
			int64(2),
			true,
		},
		{
			int64(0),
			false,
		},
	}

	for _, test := range tests {
		assert.Equal(t, test.expected, Int64ToBool(test.in))
	}
}

func TestAllInRange(t *testing.T) {
	tests := []struct {
		in       []int
		min      int
		max      int
		expected bool
	}{
		{[]int{1, 2, 3}, 0, 4, true},
		{[]int{1, 2, 3}, 1, 4, true},
		{[]int{1, 2, 3}, 1, 3, true},
		{[]int{1, 2, 3, 7}, 0, 4, false},
		{[]int{-3, 1, 2, 3}, 0, 4, false},
	}
	for _, test := range tests {
		assert.Equal(t, test.expected, AllInRange(test.in, test.min, test.max))
	}
}

func TestHasDuplicates(t *testing.T) {
	tests := []struct {
		in       []int
		expected bool
	}{
		{[]int{1, 2, 3}, false},
		{[]int{1, 3}, false},
		{[]int{1, 3, 3}, true},
		{[]int{1, 3, -3}, false},
	}
	for _, test := range tests {
		assert.Equal(t, test.expected, HasDuplicates(test.in))
	}
}

func TestOffsetArrayIfNegative(t *testing.T) {
	tests := []struct {
		in       []int
		offset   int
		expected []int
	}{
		{[]int{1, 2, 3}, 2, []int{1, 2, 3}},
		{[]int{1, 2, 3, -1}, 2, []int{1, 2, 3, 1}},
		{[]int{0, 1}, 3, []int{0, 1}},
		{[]int{-2, 2}, 3, []int{1, 2}},
	}
	for _, test := range tests {
		OffsetArrayIfNegative(test.in, test.offset)
		// The array is modified in place.
		assert.Equal(t, test.expected, test.in)
	}
}

func TestOffsetTensorIfNegative(t *testing.T) {
	tests := []struct {
		in       []int
		offset   int
		expected []int
	}{
		{[]int{1, 2, 3}, 2, []int{1, 2, 3}},
		{[]int{1, 2, 3, -1}, 2, []int{1, 2, 3, 1}},
		{[]int{0, 1}, 3, []int{0, 1}},
		{[]int{-2, 2}, 3, []int{1, 2}},
	}
	for _, test := range tests {
		tIn := tensor.New(tensor.WithShape(len(test.in)), tensor.WithBacking(test.in))
		err := OffsetTensorIfNegative(tIn, test.offset)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, tIn.Data())
	}
}

func TestAnyToIntSlice(t *testing.T) {
	tests := []struct {
		in       interface{}
		expected []int
		err      error
	}{
		{
			[]int8{1, 2, 3},
			[]int{1, 2, 3},
			nil,
		},
		{
			[]int16{1, 2, 3},
			[]int{1, 2, 3},
			nil,
		},
		{
			[]int32{1, 2, 3},
			[]int{1, 2, 3},
			nil,
		},
		{
			[]int64{1, 2, 3},
			[]int{1, 2, 3},
			nil,
		},
		{
			"some string",
			nil,
			ErrCast,
		},
	}

	for _, test := range tests {
		res, err := AnyToIntSlice(test.in)

		assert.Equal(t, test.expected, res)
		assert.Equal(t, test.err, err)
	}
}

func TestGetValueAsTensorType(t *testing.T) {
	tests := []struct {
		value         float64
		dtype         tensor.Dtype
		expectedValue interface{}
		err           error
	}{
		{
			1.0,
			tensor.Bool,
			true,
			nil,
		},
		{
			1.0,
			tensor.Int8,
			int8(1),
			nil,
		},
		{
			1.0,
			tensor.Int16,
			int16(1),
			nil,
		},
		{
			1.0,
			tensor.Int32,
			int32(1),
			nil,
		},
		{
			1.0,
			tensor.Int64,
			int64(1),
			nil,
		},
		{
			1.0,
			tensor.Float32,
			float32(1),
			nil,
		},
		{
			1.0,
			tensor.Float64,
			float64(1),
			nil,
		},
		{
			1.0,
			tensor.Complex64,
			nil,
			ErrCast,
		},
	}

	for _, test := range tests {
		val, err := GetValueAsTensorType(test.value, test.dtype)
		assert.Equal(t, err, test.err)
		assert.Equal(t, test.expectedValue, val)
	}
}

func TestIfScalarToSlice(t *testing.T) {
	tests := []struct {
		in       interface{}
		expected interface{}
	}{
		{
			int8(1),
			[]int8{1},
		},
		{
			int16(1),
			[]int16{1},
		},
		{
			int32(1),
			[]int32{1},
		},
		{
			int64(1),
			[]int64{1},
		},
		{
			int(1),
			[]int{1},
		},
		{
			float32(1),
			[]float32{1},
		},
		{
			float64(1),
			[]float64{1},
		},
		{
			complex64(1),
			[]complex64{1},
		},
		{
			complex128(1),
			[]complex128{1},
		},
		{
			uint8(1),
			uint8(1),
		},
	}

	for _, test := range tests {
		res := IfScalarToSlice(test.in)
		assert.Equal(t, test.expected, res)
	}
}

func TestZeros(t *testing.T) {
	assert.Equal(t, []float32{0, 0}, Zeros(2))
	assert.Equal(t, []float32{0, 0, 0, 0}, Zeros(4))
}

func TestFull(t *testing.T) {
	assert.Equal(t, []float32{1.0, 1.0}, Full(2, 1.0))
	assert.Equal(t, []float32{3.1, 3.1, 3.1}, Full(3, 3.1))
}

func TestOnes(t *testing.T) {
	assert.Equal(t, []float32{1.0, 1.0}, Ones(2))
	assert.Equal(t, []float32{1, 1, 1, 1, 1}, Ones(5))
}

func TestArange(t *testing.T) {
	assert.Equal(t, []float32{0, 1}, Arange(2, 1))
	assert.Equal(t, []float32{0, 0.2, 0.4, 0.6}, Arange(4, 0.2))
}

func TestNElements(t *testing.T) {
	assert.Equal(t, 2, NElements(2, 1))
	assert.Equal(t, 6, NElements(2, 3))
	assert.Equal(t, 60, NElements(2, 5, 3, 2))
}

func TestPairwiseAssign(t *testing.T) {
	tests := []struct {
		t1  tensor.Tensor
		t2  tensor.Tensor
		err error
	}{
		{
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4})),
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 1, 1, 1})),
			nil,
		},
		{
			tensor.New(tensor.WithShape(3, 1), tensor.WithBacking([]float32{1, 2, 3})),
			tensor.New(tensor.WithShape(3, 1), tensor.WithBacking([]float32{2.5, 2.1, 0.0})),
			nil,
		},
		{
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4})),
			tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{1, 1})),
			ErrInvalidShape,
		},
	}

	for _, test := range tests {
		err := PairwiseAssign(test.t1, test.t2)

		assert.Equal(t, err, test.err)

		if err == nil {
			assert.Equal(t, test.t2.Data(), test.t1.Data())
		}
	}
}
