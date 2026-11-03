package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	tests := []struct {
		tensor           *Tensor
		expectedShape    Shape
		expectedDataType DataType
		expectedData     any
	}{
		{New([]float32{1, 2, 3, 4, 5, 6}, 2, 3), Shape{2, 3}, Float32, []float32{1, 2, 3, 4, 5, 6}},
		{New([]int64{1, 2}, 2), Shape{2}, Int64, []int64{1, 2}},
		{New([]bool{true}, 1, 1), Shape{1, 1}, Bool, []bool{true}},
		{New([]uint8{}, 0, 3), Shape{0, 3}, Uint8, []uint8{}},
		{Scalar(float64(2.5)), Shape{}, Float64, []float64{2.5}},
		{Zeros(Int32, 2, 2), Shape{2, 2}, Int32, []int32{0, 0, 0, 0}},
		{Zeros(Int16), Shape{}, Int16, []int16{0}},
	}

	for _, test := range tests {
		assert.Equal(t, test.expectedShape, test.tensor.Shape())
		assert.Equal(t, test.expectedDataType, test.tensor.DataType())
		assert.Equal(t, test.expectedData, test.tensor.Data())
		assert.Equal(t, len(test.expectedShape), test.tensor.Rank())
	}
}

func TestNewShapeMismatchPanics(t *testing.T) {
	assert.Panics(t, func() { New([]float32{1, 2, 3}, 2, 2) })
	assert.Panics(t, func() { New([]float32{1, 2}) })
}

func TestNewCopiesShape(t *testing.T) {
	shape := []int{2, 2}
	x := New([]float32{1, 2, 3, 4}, shape...)

	shape[0] = 4

	assert.Equal(t, Shape{2, 2}, x.Shape())
}

func TestZerosUnsupportedDataTypePanics(t *testing.T) {
	assert.Panics(t, func() { Zeros(Undefined, 2) })
}

func TestData(t *testing.T) {
	x := New([]float32{1, 2}, 2)

	assert.Equal(t, []float32{1, 2}, Data[float32](x))
	assert.Panics(t, func() { Data[float64](x) })
}

func TestReshape(t *testing.T) {
	tests := []struct {
		shape       []int
		newShape    []int
		expectedErr bool
	}{
		{[]int{2, 3}, []int{3, 2}, false},
		{[]int{2, 3}, []int{6}, false},
		{[]int{1}, []int{}, false},
		{[]int{}, []int{1, 1}, false},
		{[]int{2, 3}, []int{4}, true},
	}

	for _, test := range tests {
		x := Zeros(Float32, test.shape...)

		reshaped, err := x.Reshape(test.newShape...)
		if test.expectedErr {
			assert.ErrorIs(t, err, ErrShapeMismatch)

			continue
		}

		assert.Nil(t, err)
		assert.Equal(t, Shape(test.newShape), reshaped.Shape())
		assert.Equal(t, Shape(test.shape), x.Shape())
	}
}

func TestReshapeSharesData(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)

	reshaped, err := x.Reshape(4)
	assert.Nil(t, err)

	Data[float32](reshaped)[0] = 10

	assert.Equal(t, []float32{10, 2, 3, 4}, x.Data())
}

func TestClone(t *testing.T) {
	x := New([]int32{1, 2, 3, 4}, 2, 2)
	clone := x.Clone()

	Data[int32](clone)[0] = 10

	assert.Equal(t, []int32{1, 2, 3, 4}, x.Data())
	assert.Equal(t, []int32{10, 2, 3, 4}, clone.Data())
	assert.Equal(t, x.Shape(), clone.Shape())
	assert.Equal(t, Int32, clone.DataType())
}

func TestDataTypeString(t *testing.T) {
	assert.Equal(t, "float32", Float32.String())
	assert.Equal(t, "bool", Bool.String())
	assert.Equal(t, "undefined", DataType(8).String())
	assert.True(t, Uint64.IsSupported())
	assert.False(t, DataType(8).IsSupported())
}

func TestShape(t *testing.T) {
	assert.Equal(t, 6, Shape{2, 3}.Size())
	assert.Equal(t, 1, Shape{}.Size())
	assert.Equal(t, 0, Shape{2, 0}.Size())
	assert.True(t, Shape{2, 3}.Equal([]int{2, 3}))
	assert.False(t, Shape{2, 3}.Equal(Shape{3, 2}))
	assert.False(t, Shape{2}.Equal(Shape{2, 1}))
}
