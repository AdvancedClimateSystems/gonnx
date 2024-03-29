package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAcoshInit(t *testing.T) {
	c := &Acosh{}

	// since 'acosh' does not have any attributes we pass in nil. This should not
	// fail initializing the acosh.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestAcosh(t *testing.T) {
	tests := []struct {
		acosh    *Acosh
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Acosh{},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{0, 1.316958, 1.7627472, 2.063437},
		},
		{
			&Acosh{},
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{0, 1.316958, 1.7627472, 2.063437},
		},
		{
			&Acosh{},
			[]float32{2, 2, 2, 2},
			[]int{1, 4},
			[]float32{1.316958, 1.316958, 1.316958, 1.316958},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.acosh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAcosh(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, &Acosh{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Acosh{}),
		},
	}

	for _, test := range tests {
		acosh := &Acosh{}
		validated, err := acosh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
