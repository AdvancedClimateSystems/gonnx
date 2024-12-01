package atan

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAtan7Init(t *testing.T) {
	a := &Atan7{}

	// since 'atan' does not have any attributes we pass in nil. This should not
	// fail initializing the atan.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAtan7(t *testing.T) {
	tests := []struct {
		atan     *Atan7
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Atan7{},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{0.7853982, 1.1071488, 1.2490457, 1.3258177},
		},
		{
			&Atan7{},
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{0.7853982, 1.1071488, 1.2490457, 1.3258177},
		},
		{
			&Atan7{},
			[]float32{2, 2, 2, 2},
			[]int{1, 4},
			[]float32{1.1071488, 1.1071488, 1.1071488, 1.1071488},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.atan.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAtan7(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, &Atan7{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Atan7{}),
		},
	}

	for _, test := range tests {
		atan := &Atan7{}
		validated, err := atan.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
