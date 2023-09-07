package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAtanInit(t *testing.T) {
	a := &Atan{}

	// since 'atan' does not have any attributes we pass in nil. This should not
	// fail initializing the atan.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAtan(t *testing.T) {
	tests := []struct {
		atan    *Atan
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Atan{},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
                        []float32{0.7853982, 1.1071488, 1.2490457, 1.3258177},
		},
		{
			&Atan{},
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
                        []float32{0.7853982, 1.1071488, 1.2490457, 1.3258177},
		},
		{
			&Atan{},
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

func TestInputValidationAtan(t *testing.T) {
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
			fmt.Errorf("atan operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("atan operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		atan := &Atan{}
		validated, err := atan.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
