package relu

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestRelu13Init(t *testing.T) {
	r := &Relu13{}

	// since the relu does not have any attributes we pass in nil. This should not
	// fail initializing the relu.
	err := r.Init(nil)
	assert.Nil(t, err)
}

func TestRelu13(t *testing.T) {
	tests := []struct {
		relu     *Relu13
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Relu13{},
			[]float32{-4, -4, -4, -3, -2, -1},
			[]int{3, 2},
			[]float32{0, 0, 0, 0, 0, 0},
		},
		{
			&Relu13{},
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{0, 0, 0, 3, 2, 1},
		},
		{
			&Relu13{},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{4, 3},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{ops.TensorWithBackingFixture(test.backing, test.shape...)}
		res, err := test.relu.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationRelu13(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, &Relu13{}),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", &Relu13{}),
		},
	}

	for _, test := range tests {
		relu := &Relu13{}
		validated, err := relu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
