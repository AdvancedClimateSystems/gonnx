package sinh

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSinh9Init(t *testing.T) {
	s := &Sinh9{}

	// since 'sinh' does not have any attributes we pass in nil. This should not
	// fail initializing the sinh.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSinh9(t *testing.T) {
	tests := []struct {
		sinh     *Sinh9
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Sinh9{},
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
			[]float32{-3.6268604, -1.1752012, 0, 1.1752012},
		},
		{
			&Sinh9{},
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{1.1752012, 10.017875, 27.289917, 74.20321},
		},
		{
			&Sinh9{},
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{-1.1752012, -1.1752012, -1.1752012, -1.1752012},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.sinh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationSinh9(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, &Sinh9{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Sinh9{}),
		},
	}

	for _, test := range tests {
		sinh := &Sinh9{}
		validated, err := sinh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
