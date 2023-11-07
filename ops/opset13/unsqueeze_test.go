package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestUnsqueezeInit(t *testing.T) {
	s := &Unsqueeze{}

	// since the unsqueeze does not have any attributes we pass in nil. This should not
	// fail initializing the unsqueeze.
	err := s.Init(nil)
	assert.NoError(t, err)
}

func TestAxesOutRangeError(t *testing.T) {
	op := &Unsqueeze{}
	err := op.Init(nil)
	assert.Nil(t, err)

	axes := []int64{4}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	expected := ops.ErrNotAllAxesInRange(3, 3)
	assert.Equal(t, err, expected)
}

func TestDuplicateEntriesAfterOffsetNotAllowed(t *testing.T) {
	op := &Unsqueeze{}
	err := op.Init(nil)
	assert.Nil(t, err)

	// -1 will be offset to 3 (since outputrank = 4)
	axes := []int64{3, -1}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	assert.EqualError(t, err, "invalid input tensor for unsqueeze operator: axes cannot have duplicate entries after offset")
}

func TestDuplicateEntriesNotAllowed(t *testing.T) {
	op := &Unsqueeze{}
	err := op.Init(nil)
	assert.Nil(t, err)

	axes := []int64{0, 0}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	assert.EqualError(t, err, "invalid input tensor for unsqueeze operator: axes cannot have duplicate entries after offset")
}

func TestUnsqueeze(t *testing.T) {
	tests := []struct {
		data              interface{}
		dataShape         []int
		axes              []int64
		expectOutputShape []int
	}{
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{0}, []int{1, 2, 2}},
		{[]int64{1}, []int{1}, []int64{1}, []int{1, 1}},
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{0, -1}, []int{1, 2, 2, 1}},
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{-1, 0}, []int{1, 2, 2, 1}},

		{
			[]int16{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{0, 2, 4, 6},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]complex128{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{6, 0, 4, 2},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{-7, -5, -3, -1},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{-1, -7, -3, -5},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{0, 1, 2, 3},
			[]int{1, 1, 1, 1, 2, 2, 2},
		},
	}
	for _, test := range tests {
		op := &Unsqueeze{}
		err := op.Init(nil)
		assert.Nil(t, err)

		axes := test.axes
		data := test.data
		dataIn := ops.TensorWithBackingFixture(data, test.dataShape...)
		axesIn := ops.TensorWithBackingFixture(axes, len(axes))

		res, err := op.Apply([]tensor.Tensor{dataIn, axesIn})
		assert.NoError(t, err)

		shape := res[0].Shape()
		expShape := tensor.Shape(test.expectOutputShape)
		assert.Equal(t, expShape, shape)
	}
}

func TestInputValidationUnsqueeze(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputCount(1, &Unsqueeze{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Unsqueeze{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			ops.ErrInvalidInputType(1, "int32", &Unsqueeze{}),
		},
	}

	for _, test := range tests {
		unsqueeze := &Unsqueeze{}
		validated, err := unsqueeze.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
