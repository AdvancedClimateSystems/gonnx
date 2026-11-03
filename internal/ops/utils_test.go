package ops

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
