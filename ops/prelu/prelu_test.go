package prelu

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestPReluInit(t *testing.T) {
	tests := []struct {
		version int64
		err     error
	}{
		{7, nil},
		{9, nil},
	}

	for _, test := range tests {
		r := preluVersions[test.version]()
		err := r.Init(nil)
		assert.Equal(t, test.err, err)
	}
}

func TestPRelu(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		slope    []float32
		shape    []int
		expected []float32
	}{
		{
			7,
			[]float32{-4, -4, -4, -3, -2, -1},
			[]float32{2, 2, 4, 4, 0, 0},
			[]int{3, 2},
			[]float32{-8, -8, -16, -12, 0, 0},
		},
		{
			9,
			[]float32{-4, -4, -4, 3, 2, 1},
			[]float32{2, 2, 4, 4, 0, 0},
			[]int{3, 2},
			[]float32{-8, -8, -16, 3, 2, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
			ops.TensorWithBackingFixture(test.slope, test.shape...),
		}
		prelu := preluVersions[test.version]()
		res, err := prelu.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationPRelu(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, prelu7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int32", prelu7BaseOpFixture()),
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", prelu9BaseOpFixture()),
		},
	}

	for _, test := range tests {
		prelu := preluVersions[test.version]()
		validated, err := prelu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func BenchmarkPRelu_Apply(b *testing.B) {
	prelu := &PRelu{}
	input := ops.Float32TensorFixture(3, 256, 256)
	slope := ops.Float32TensorFixture(3, 256, 256)
	inputs := []tensor.Tensor{input, slope}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		y, err := prelu.Apply(inputs)
		if err != nil {
			b.Fatal(err)
		}

		_ = y
	}
}

func prelu7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 2, 2, prelu7TypeConstraints, "prelu")
}

func prelu9BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(9, 2, 2, preluTypeConstraints, "prelu")
}
