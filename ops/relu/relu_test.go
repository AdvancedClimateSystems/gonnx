package relu

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReluInit(t *testing.T) {
	tests := []struct {
		version int64
		err     error
	}{
		{6, nil},
		{13, nil},
	}

	for _, test := range tests {
		r := reluVersions[test.version]()
		err := r.Init(nil)
		assert.Equal(t, test.err, err)
	}
}

func TestRelu(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			6,
			[]float32{-4, -4, -4, -3, -2, -1},
			[]int{3, 2},
			[]float32{0, 0, 0, 0, 0, 0},
		},
		{
			13,
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{0, 0, 0, 3, 2, 1},
		},
		{
			13,
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{4, 3},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{ops.TensorWithBackingFixture(test.backing, test.shape...)}

		relu := reluVersions[test.version]()
		res, err := relu.Apply(inputs)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationRelu(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			6,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			6,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			6,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, relu6BaseOpFixture()),
		},
		{
			6,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", relu6BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, relu13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", relu13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		relu := reluVersions[test.version]()

		validated, err := relu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func relu6BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(6, 1, 1, reluTypeConstraints, "relu")
}

func relu13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, reluTypeConstraints, "relu")
}
