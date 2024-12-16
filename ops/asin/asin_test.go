package asin

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAsinInit(t *testing.T) {
	s := &Asin{}

	// since 'asin' does not have any attributes we pass in nil. This should not
	// fail initializing the asin.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestAsin(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			7,
			[]float32{-1, -1, 0, 1},
			[]int{2, 2},
			[]float32{-1.5707964, -1.5707964, 0, 1.5707964},
		},
		{
			7,
			[]float32{1, 0.5, 0.0, -0.5},
			[]int{1, 4},
			[]float32{1.5707964, 0.5235988, 0, -0.5235988},
		},
		{
			7,
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{-1.5707964, -1.5707964, -1.5707964, -1.5707964},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		asin := asinVersions[test.version]()

		res, err := asin.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAsin(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, asin7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", asin7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		asin := asinVersions[test.version]()
		validated, err := asin.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func asin7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 1, 1, asinTypeConstraints, "asin")
}
