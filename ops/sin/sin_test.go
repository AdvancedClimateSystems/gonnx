package sin

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSinInit(t *testing.T) {
	a := &Sin{}

	// since 'sin' does not have any attributes we pass in nil. This should not
	// fail initializing the sin.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestSin(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			7,
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
			[]float32{-0.9092974, -0.84147096, 0, 0.84147096},
		},
		{
			7,
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{0.84147096, 0.14112, -0.7568025, -0.9589243},
		},
		{
			7,
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{-0.84147096, -0.84147096, -0.84147096, -0.84147096},
		},
	}

	for _, test := range tests {
		sin := sinVersions[test.version]()
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := sin.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationSin(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, sin7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", sin7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		sin := sinVersions[test.version]()
		validated, err := sin.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func sin7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 1, 1, sinTypeConstraints, "sin")
}
