package acosh

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAcosh9Init(t *testing.T) {
	c := &Acosh{}

	// since 'acosh' does not have any attributes we pass in nil. This should not
	// fail initializing the acosh.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestAcosh9(t *testing.T) {
	tests := []struct {
		acosh    ops.Operator
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			newAcosh(),
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{0, 1.316958, 1.7627472, 2.063437},
		},
		{
			newAcosh(),
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{0, 1.316958, 1.7627472, 2.063437},
		},
		{
			newAcosh(),
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
			ops.ErrInvalidInputCount(0, ops.NewBaseOperator(9, 1, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}, "acosh")),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", ops.NewBaseOperator(9, 1, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}, "acosh")),
		},
	}

	for _, test := range tests {
		acosh := newAcosh()
		validated, err := acosh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
