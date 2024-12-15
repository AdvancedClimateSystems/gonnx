package mul

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestMulInit(t *testing.T) {
	m := &Mul{}

	// since 'mul' does not have any attributes we pass in nil. This should not
	// fail initializing the mul.
	err := m.Init(nil)
	assert.Nil(t, err)
}

func TestMul(t *testing.T) {
	tests := []struct {
		version  int64
		backings [][]float32
		shapes   [][]int
		expected []float32
	}{
		{
			13,
			[][]float32{{0, 1, 2, 3}, {1, 1, 1, 1}},
			[][]int{{2, 2}, {2, 2}},
			[]float32{0, 1, 2, 3},
		},
		{
			13,
			[][]float32{{0, 1, 2, 3, 4, 5}, {2, 2, 2, 2, 2, 2}},
			[][]int{{3, 2}, {3, 2}},
			[]float32{0, 2, 4, 6, 8, 10},
		},
		{
			13,
			[][]float32{{0, 1}, {0, 1, 2, 3}},
			[][]int{{2}, {2, 2}},
			[]float32{0, 1, 0, 3},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		mul := mulVersions[test.version]()
		res, err := mul.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestMulFail(t *testing.T) {
	inputs := []tensor.Tensor{
		ops.TensorWithBackingFixture([]float32{1, 2, 3, 4}, 2, 2),
		ops.TensorWithBackingFixture([]float32{1, 2, 3}, 3),
	}

	mul := &Mul{}
	_, err := mul.Apply(inputs)
	assert.Equal(
		t,
		err,
		ops.ErrMultidirBroadcast([]int{2, 2}, []int{3}, ops.ErrIncompatibleDimensions()),
	)
}

func TestInputValidationMul(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint64{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, mul7BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, mul13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int", mul13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		mul := mulVersions[test.version]()
		validated, err := mul.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func mul7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 2, 2, mulTypeConstraints, "mul")
}

func mul13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 2, mulTypeConstraints, "mul")
}
