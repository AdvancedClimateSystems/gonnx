package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCosInit(t *testing.T) {
	c := &Cos{}

	// since 'cos' does not have any attributes we pass in nil. This should not
	// fail initializing the cos.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestCos(t *testing.T) {
	tests := []struct {
		cos      *Cos
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Cos{},
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
                        []float32{-0.41614684, 0.5403023, 1, 0.5403023},
		},
		{
			&Cos{},
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
                        []float32{0.5403023, -0.9899925, -0.6536436, 0.2836622},
		},
		{
			&Cos{},
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
                        []float32{0.5403023, 0.5403023, 0.5403023, 0.5403023},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.cos.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationCos(t *testing.T) {
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
			fmt.Errorf("cos operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("cos operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		cos := &Cos{}
		validated, err := cos.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
