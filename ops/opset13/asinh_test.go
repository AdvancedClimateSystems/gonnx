package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAsinhInit(t *testing.T) {
	c := &Asinh{}

	// since 'asinh' does not have any attributes we pass in nil. This should not
	// fail initializing the asinh.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestAsinh(t *testing.T) {
	tests := []struct {
		asinh    *Asinh
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Asinh{},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{0.8813736, 1.4436355, 1.8184465, 2.0947125},
		},
		{
			&Asinh{},
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{0.8813736, 1.4436355, 1.8184465, 2.0947125},
		},
		{
			&Asinh{},
			[]float32{2, 2, 2, 2},
			[]int{1, 4},
			[]float32{1.4436355, 1.4436355, 1.4436355, 1.4436355},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.asinh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAsinh(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, &Asinh{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Asinh{}),
		},
	}

	for _, test := range tests {
		asinh := &Asinh{}
		validated, err := asinh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
