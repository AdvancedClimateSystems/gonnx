package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestMatMulInit(t *testing.T) {
	s := newMatMul()
	// since the matMul does not have any attributes we expect it to initialize even
	// when nil is passed.
	err := s.Init(nil)

	assert.Nil(t, err)
}

func TestMatMul(t *testing.T) {
	tests := []struct {
		backings      [][]float32
		shapes        [][]int
		expected      []float32
		expectedShape tensor.Shape
	}{
		{
			[][]float32{{3, 1, 4}, {4, 3, 2, 5, 6, 8}},
			[][]int{{1, 3}, {3, 2}},
			[]float32{38, 46},
			[]int{1, 2},
		},
		{
			[][]float32{{3, 4, 7, 2, 5, 9}, {3, 1, 5, 6, 9, 7}},
			[][]int{{3, 2}, {2, 3}},
			[]float32{33, 39, 43, 33, 25, 49, 69, 86, 88},
			[]int{3, 3},
		},
		{
			[][]float32{
				{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
				{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			},
			[][]int{{2, 2, 3}, {2, 3, 2}},
			[]float32{13, 16, 40, 52, 193, 214, 274, 304},
			[]int{2, 2, 2},
		},
		{
			[][]float32{
				{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
				{1, 2, 3, 4, 5, 6},
			},
			[][]int{{2, 2, 3}, {3, 2}},
			[]float32{13, 16, 40, 52, 67, 88, 94, 124},
			[]int{2, 2, 2},
		},
		{
			[][]float32{
				{0, 1, 2, 3, 4, 5},
				{1, 2, 3, 4},
			},
			[][]int{{2, 3, 1}, {1, 4}},
			[]float32{0, 0, 0, 0, 1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16, 5, 10, 15, 20},
			[]int{2, 3, 4},
		},
		{
			[][]float32{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {1, 2}},
			[][]int{{2, 2, 2, 2}, {2}},
			[]float32{5, 11, 17, 23, 29, 35, 41, 47},
			[]int{2, 2, 2},
		},
		{
			[][]float32{{1, 2}, {1, 2, 3, 4}},
			[][]int{{2}, {2, 2}},
			[]float32{7, 10},
			[]int{2},
		},
		{
			[][]float32{{1, 2, 3, 4}, {1, 2}},
			[][]int{{2, 2}, {2}},
			[]float32{5, 11},
			[]int{2},
		},
	}

	for _, test := range tests {
		matmul := &MatMul{}
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		res, err := matmul.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
		assert.Equal(t, test.expectedShape, res[0].Shape())
	}
}

func TestBroadcastTensors(t *testing.T) {
	tests := []struct {
		shapes         [][]int
		expectedShapes []tensor.Shape
		err            error
	}{
		{
			[][]int{{2, 2, 3}, {3, 2}},
			[]tensor.Shape{{2, 2, 3}, {2, 3, 2}},
			nil,
		},
		{
			[][]int{{10, 5}, {20, 8, 5}},
			[]tensor.Shape{{20, 10, 5}, {20, 8, 5}},
			nil,
		},
	}

	for _, test := range tests {
		matmul := &MatMul{}
		A := ops.Float32TensorFixture(test.shapes[0]...)
		B := ops.Float32TensorFixture(test.shapes[1]...)
		newA, newB, err := matmul.broadcastTensors(A, B)

		assert.Equal(t, test.err, err)
		assert.Equal(t, test.expectedShapes[0], newA.Shape())
		assert.Equal(t, test.expectedShapes[1], newB.Shape())
	}
}

func TestInputValidationMatMul(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("matmul operator: expected 2 input tensors, got 1"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			fmt.Errorf("matmul operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		matmul := &MatMul{}
		validated, err := matmul.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
