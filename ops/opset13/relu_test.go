package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReluInit(t *testing.T) {
	r := &Relu{}

	// since the relu does not have any attributes we pass in nil. This should not
	// fail initializing the relu.
	err := r.Init(nil)
	assert.Nil(t, err)
}

func TestRelu(t *testing.T) {
	tests := []struct {
		relu     *Relu
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Relu{},
			[]float32{-4, -4, -4, -3, -2, -1},
			[]int{3, 2},
			[]float32{0, 0, 0, 0, 0, 0},
		},
		{
			&Relu{},
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{0, 0, 0, 3, 2, 1},
		},
		{
			&Relu{},
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

func TestInputValidationRelu(t *testing.T) {
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
			fmt.Errorf("relu operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("relu operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		relu := &Relu{}
		validated, err := relu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
