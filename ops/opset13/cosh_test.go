package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCoshInit(t *testing.T) {
	c := &Cosh{}

	// since 'cosh' does not have any attributes we pass in nil. This should not
	// fail initializing the cosh.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestCosh(t *testing.T) {
	tests := []struct {
		cosh     *Cosh
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Cosh{},
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
			[]float32{3.7621956, 1.5430807, 1, 1.5430807},
		},
		{
			&Cosh{},
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{1.5430807, 10.067662, 27.308233, 74.209946},
		},
		{
			&Cosh{},
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{1.5430807, 1.5430807, 1.5430807, 1.5430807},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.cosh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationCosh(t *testing.T) {
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
			fmt.Errorf("cosh operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("cosh operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		cosh := &Cosh{}
		validated, err := cosh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
