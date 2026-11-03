package ops

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/tensor"
	"github.com/stretchr/testify/assert"
)

func TestReduce(t *testing.T) {
	tests := []struct {
		reduction     Reduction
		axes          []int
		keepDims      bool
		expectedShape tensor.Shape
		expected      []float32
	}{
		{ReduceSum, []int{0}, false, tensor.Shape{3}, []float32{3, 5, 7}},
		{ReduceSum, []int{-1}, true, tensor.Shape{2, 1}, []float32{3, 12}},
		{ReduceSum, nil, false, tensor.Shape{}, []float32{15}},
		{ReduceSum, nil, true, tensor.Shape{1, 1}, []float32{15}},
		{ReduceMean, []int{1}, false, tensor.Shape{2}, []float32{1, 4}},
		{ReduceMax, []int{0}, true, tensor.Shape{1, 3}, []float32{3, 4, 5}},
		{ReduceMin, []int{0, 1}, false, tensor.Shape{}, []float32{0}},
	}

	for _, test := range tests {
		out, err := Reduce(Float32TensorFixture(2, 3), test.reduction, test.axes, test.keepDims)

		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}
}

func TestReduceMiddleAxis(t *testing.T) {
	out, err := Reduce(tensor.New([]int32{5, -1, 2, 7, 3, 3, -4, 0}, 2, 2, 2), ReduceMax, []int{1}, false)

	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{2, 2}, out.Shape())
	assert.Equal(t, []int32{5, 7, 3, 3}, out.Data())
}

func TestReduceInvalidAxis(t *testing.T) {
	_, err := Reduce(Float32TensorFixture(2, 3), ReduceSum, []int{2}, false)
	assert.ErrorIs(t, err, ErrAxisNotInRange)
}

func TestArgMax(t *testing.T) {
	x := tensor.New([]float32{1, 5, 5, 2, 0, 3}, 2, 3)

	out, err := ArgMax(x, 1, true)
	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{2, 1}, out.Shape())
	assert.Equal(t, []int64{1, 2}, out.Data())

	out, err = ArgMax(x, 0, false)
	assert.Nil(t, err)
	assert.Equal(t, tensor.Shape{3}, out.Shape())
	assert.Equal(t, []int64{1, 0, 0}, out.Data())
}

func TestCumSum(t *testing.T) {
	tests := []struct {
		axis      int
		exclusive bool
		reverse   bool
		expected  []int64
	}{
		{0, false, false, []int64{1, 2, 3, 5, 7, 9}},
		{1, false, false, []int64{1, 3, 6, 4, 9, 15}},
		{-1, true, false, []int64{0, 1, 3, 0, 4, 9}},
		{1, false, true, []int64{6, 5, 3, 15, 11, 6}},
		{1, true, true, []int64{5, 3, 0, 11, 6, 0}},
	}

	for _, test := range tests {
		out, err := CumSum(tensor.New([]int64{1, 2, 3, 4, 5, 6}, 2, 3), test.axis, test.exclusive, test.reverse)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, out.Data())
	}
}

func TestSoftmax(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 1000, 1000, 1000}, 2, 3)

	out, err := Softmax(x, 1, false)
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []float64{0.09003057, 0.24472847, 0.66524096, 1.0 / 3, 1.0 / 3, 1.0 / 3}, out.Data(), 1e-8)

	out, err = Softmax(x, 1, true)
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []float64{-2.40760596, -1.40760596, -0.40760596, -1.09861229, -1.09861229, -1.09861229}, out.Data(), 1e-8)
}
