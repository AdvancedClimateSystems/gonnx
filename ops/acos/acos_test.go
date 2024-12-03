package acos

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAcosInit(t *testing.T) {
	c := &Acos{}

	// since 'acos' does not have any attributes we pass in nil. This should not
	// fail initializing the acos.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestAcos(t *testing.T) {
	tests := []struct {
		acos     ops.Operator 
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			newAcos(),
			[]float32{-1, -1, 0, 1},
			[]int{2, 2},
			[]float32{3.1415927, 3.1415927, 1.5707964, 0},
		},
		{
			newAcos(),
			[]float32{1, 0.5, 0.0, -0.5},
			[]int{1, 4},
			[]float32{0, 1.0471976, 1.5707964, 2.0943952},
		},
		{
			newAcos(),
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{3.1415927, 3.1415927, 3.1415927, 3.1415927},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.acos.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAcos(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, ops.NewBaseOperator(7, 1, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}, "acos")),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", ops.NewBaseOperator(7, 1, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}, "acos")),
		},
	}

	for _, test := range tests {
		acos := newAcos()
		validated, err := acos.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
