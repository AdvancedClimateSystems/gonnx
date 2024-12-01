package asin

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAsin7Init(t *testing.T) {
	s := &Asin7{}

	// since 'asin' does not have any attributes we pass in nil. This should not
	// fail initializing the asin.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestAsin7(t *testing.T) {
	tests := []struct {
		asin     *Asin7
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Asin7{},
			[]float32{-1, -1, 0, 1},
			[]int{2, 2},
			[]float32{-1.5707964, -1.5707964, 0, 1.5707964},
		},
		{
			&Asin7{},
			[]float32{1, 0.5, 0.0, -0.5},
			[]int{1, 4},
			[]float32{1.5707964, 0.5235988, 0, -0.5235988},
		},
		{
			&Asin7{},
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{-1.5707964, -1.5707964, -1.5707964, -1.5707964},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.asin.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAsin7(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
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
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, &Asin7{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Asin7{}),
		},
	}

	for _, test := range tests {
		asin := &Asin7{}
		validated, err := asin.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
