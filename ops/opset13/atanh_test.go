package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAtanhInit(t *testing.T) {
	a := &Atanh{}

	// since 'atanh' does not have any attributes we pass in nil. This should not
	// fail initializing the atanh.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAtanh(t *testing.T) {
	tests := []struct {
		atanh    *Atanh
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Atanh{},
			[]float32{-0.9, -0.5, 0, 0.5},
			[]int{2, 2},
			[]float32{-1.4722193, -0.54930615, 0, 0.54930615},
		},
		{
			&Atanh{},
			[]float32{-0.9, -0.5, 0, 0.5},
			[]int{1, 4},
			[]float32{-1.4722193, -0.54930615, 0, 0.54930615},
		},
		{
			&Atanh{},
			[]float32{0.5, 0.5, 0.5, 0.5},
			[]int{1, 4},
			[]float32{0.54930615, 0.54930615, 0.54930615, 0.54930615},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.atanh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAtanh(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, &Atanh{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Atanh{}),
		},
	}

	for _, test := range tests {
		atanh := &Atanh{}
		validated, err := atanh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
