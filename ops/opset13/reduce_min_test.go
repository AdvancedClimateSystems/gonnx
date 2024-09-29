package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReduceMinInit(t *testing.T) {
	r := &ReduceMin{}
	err := r.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axes", Ints: []int64{1, 3}},
			{Name: "keepdims", I: 0},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, []int{1, 3}, r.axes)
	assert.Equal(t, false, r.keepDims)
}

func TestReduceMin(t *testing.T) {
	tests := []struct {
		reduceMin       *ReduceMin
		backing         []float32
		shape           []int
		expectedBacking []float32
		expectedShape   tensor.Shape
	}{
		{
			&ReduceMin{axes: []int{0}, keepDims: false},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 1},
			[]int{2},
		},
		{
			&ReduceMin{axes: []int{0}, keepDims: true},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 1},
			[]int{1, 2},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: false},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 2},
			[]int{2},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: true},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 2},
			[]int{2, 1},
		},
		{
			&ReduceMin{axes: []int{0}, keepDims: false},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{0, 1, 2},
			[]int{3},
		},
		{
			&ReduceMin{axes: []int{0}, keepDims: true},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{0, 1, 2},
			[]int{1, 3},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: false},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{0, 3},
			[]int{2},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: true},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{0, 3},
			[]int{2, 1},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: false},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 1, 2, 6, 7, 8},
			[]int{2, 3},
		},
		{
			&ReduceMin{axes: []int{1}, keepDims: true},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 1, 2, 6, 7, 8},
			[]int{2, 1, 3},
		},
		{
			&ReduceMin{axes: []int{0, 1}, keepDims: false},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 1, 2},
			[]int{3},
		},
		{
			&ReduceMin{axes: []int{0, 1}, keepDims: true},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 1, 2},
			[]int{1, 1, 3},
		},
		{
			&ReduceMin{axes: []int{1, 2}, keepDims: false},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 6},
			[]int{2},
		},
		{
			&ReduceMin{axes: []int{1, 2}, keepDims: true},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{0, 6},
			[]int{2, 1, 1},
		},
		{
			&ReduceMin{axes: []int{-1}, keepDims: true},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 2},
			[]int{2, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.reduceMin.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationReduceMin(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputCount(2, &ReduceMin{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &ReduceMin{}),
		},
	}

	for _, test := range tests {
		reduceMin := &ReduceMin{}
		validated, err := reduceMin.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
