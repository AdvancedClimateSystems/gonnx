package expand

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestExpandInit(t *testing.T) {
	e := &Expand{}

	err := e.Init(nil)
	assert.Nil(t, err)
}

func TestExpand(t *testing.T) {
	tests := []struct {
		version         int64
		backing         []float32
		shape           []int
		newShapeBacking []int64
		expectedShape   tensor.Shape
		expectedData    []float32
	}{
		{
			13,
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int64{1, 1, 1},
			[]int{1, 2, 2},
			[]float32{0, 1, 2, 3},
		},
		{
			13,
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

		expand := expandVersions[test.version]()

		res, err := expand.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedData, res[0].Data())
	}
}

func TestInputValidationExpand(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(3, expand13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 1, 1}, 3),
			},
			ops.ErrInvalidInputType(0, "int", expand13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		expand := expandVersions[test.version]()
		validated, err := expand.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func expand13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 2, expandTypeConstraints, "expand")
}
