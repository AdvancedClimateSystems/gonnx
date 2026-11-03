package ops

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/tensor"
	"github.com/stretchr/testify/assert"
)

func TestTranspose(t *testing.T) {
	tests := []struct {
		shape         []int
		perm          []int
		expectedShape tensor.Shape
		expected      []float32
	}{
		{[]int{2, 3}, nil, tensor.Shape{3, 2}, []float32{0, 3, 1, 4, 2, 5}},
		{[]int{2, 3}, []int{0, 1}, tensor.Shape{2, 3}, []float32{0, 1, 2, 3, 4, 5}},
		{
			[]int{2, 2, 3}, []int{1, 0, 2},
			tensor.Shape{2, 2, 3},
			[]float32{0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11},
		},
		{
			[]int{2, 1, 3}, []int{2, 0, 1},
			tensor.Shape{3, 2, 1},
			[]float32{0, 3, 1, 4, 2, 5},
		},
	}

	for _, test := range tests {
		out, err := Transpose(Float32TensorFixture(test.shape...), test.perm...)

		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}
}

func TestTransposeInvalidPermutation(t *testing.T) {
	_, err := Transpose(Float32TensorFixture(2, 3), 0, 0)
	assert.Error(t, err)
}

func TestSlice(t *testing.T) {
	tests := []struct {
		shape         []int
		starts        []int
		ends          []int
		axes          []int
		steps         []int
		expectedShape tensor.Shape
		expected      []float32
	}{
		{[]int{2, 3}, []int{1}, []int{2}, []int{0}, []int{1}, tensor.Shape{1, 3}, []float32{3, 4, 5}},
		{[]int{2, 3}, []int{-2}, []int{100}, []int{-1}, []int{1}, tensor.Shape{2, 2}, []float32{1, 2, 4, 5}},
		{[]int{5}, []int{-1}, []int{-100}, []int{0}, []int{-2}, tensor.Shape{3}, []float32{4, 2, 0}},
		{[]int{5}, []int{4}, []int{1}, []int{0}, []int{-1}, tensor.Shape{3}, []float32{4, 3, 2}},
		{[]int{5}, []int{3}, []int{1}, []int{0}, []int{1}, tensor.Shape{0}, []float32{}},
		{
			[]int{3, 4}, []int{0, 3}, []int{3, 0}, []int{0, 1}, []int{2, -2},
			tensor.Shape{2, 2},
			[]float32{3, 1, 11, 9},
		},
	}

	for _, test := range tests {
		out, err := Slice(Float32TensorFixture(test.shape...), test.starts, test.ends, test.axes, test.steps)

		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}
}

func TestSliceZeroStep(t *testing.T) {
	_, err := Slice(Float32TensorFixture(3), []int{0}, []int{3}, []int{0}, []int{0})
	assert.Error(t, err)
}

func TestConcat(t *testing.T) {
	A := tensor.New([]int64{1, 2, 3, 4}, 2, 2)
	B := tensor.New([]int64{5, 6}, 2, 1)

	out, err := Concat(-1, A, B)
	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{2, 3}, out.Shape())
	assert.Equal(t, []int64{1, 2, 5, 3, 4, 6}, out.Data())

	_, err = Concat(0, A, B)
	assert.Equal(t, ErrIncompatibleDimensions(), err)

	_, err = Concat(0, A, tensor.New([]int32{1, 2}, 1, 2))
	assert.ErrorIs(t, err, ErrUnsupportedType)
}

func TestGather(t *testing.T) {
	data := Float32TensorFixture(3, 2)

	out, err := Gather(data, []int{2, 0, 2, 1}, tensor.Shape{2, 2}, 0)
	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{2, 2, 2}, out.Shape())
	assert.Equal(t, []float32{4, 5, 0, 1, 4, 5, 2, 3}, out.Data())

	out, err = Gather(data, []int{1}, tensor.Shape{}, 1)
	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{3}, out.Shape())
	assert.Equal(t, []float32{1, 3, 5}, out.Data())
}

func TestBroadcastTo(t *testing.T) {
	tests := []struct {
		t        *tensor.Tensor
		shape    tensor.Shape
		expected any
	}{
		{tensor.New([]float32{1, 2}, 2), tensor.Shape{2, 2}, []float32{1, 2, 1, 2}},
		{tensor.New([]float32{1, 2}, 2, 1), tensor.Shape{2, 3}, []float32{1, 1, 1, 2, 2, 2}},
		{tensor.Scalar(true), tensor.Shape{3}, []bool{true, true, true}},
		{tensor.New([]int8{1, 2}, 1, 2), tensor.Shape{2, 1, 2}, []int8{1, 2, 1, 2}},
	}

	for _, test := range tests {
		out, err := BroadcastTo(test.t, test.shape)

		assert.Nil(t, err)
		assert.Equal(t, test.shape, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}

	_, err := BroadcastTo(tensor.New([]float32{1, 2}, 2), tensor.Shape{3})
	assert.Error(t, err)
}

func TestElementwiseBroadcast(t *testing.T) {
	A := Float32TensorFixture(2, 1, 3)
	B := tensor.New([]float32{10, 20}, 2, 1)

	out, err := Add(A, B)

	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{2, 2, 3}, out.Shape())
	assert.Equal(t, []float32{10, 11, 12, 20, 21, 22, 13, 14, 15, 23, 24, 25}, out.Data())
}

func TestCoalesce(t *testing.T) {
	shape, strides := coalesce([]int{2, 1, 3, 4}, []int{12, 12, 4, 1}, []int{0, 0, 4, 1})

	assert.Equal(t, []int{2, 12}, shape)
	assert.Equal(t, [][]int{{12, 1}, {0, 1}}, strides)
}
