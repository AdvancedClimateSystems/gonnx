package expand

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestExpand13Init(t *testing.T) {
	e := &Expand13{}

	err := e.Init(nil)
	assert.Nil(t, err)
}

func TestExpand13(t *testing.T) {
	tests := []struct {
		expand          *Expand13
		backing         []float32
		shape           []int
		newShapeBacking []int64
		expectedShape   tensor.Shape
		expectedData    []float32
	}{
		{
			&Expand13{},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int64{1, 1, 1},
			[]int{1, 2, 2},
			[]float32{0, 1, 2, 3},
		},
		{
			&Expand13{},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int64{1, 3, 1, 1},
			[]int{1, 3, 2, 2},
			[]float32{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
			ops.TensorWithBackingFixture(test.newShapeBacking, len(test.newShapeBacking)),
		}

		res, err := test.expand.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedData, res[0].Data())
	}
}

func TestInputValidationExpand13(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(3, &Expand13{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			ops.ErrInvalidInputType(0, "int", &Expand13{}),
		},
	}

	for _, test := range tests {
		expand := &Expand13{}
		validated, err := expand.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
