package sqrt

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSqrtInit(t *testing.T) {
	s := &Sqrt{}
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSqrt(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			13,
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{1, 1.4142135, 1.7320508, 2},
		},
		{
			6,
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{1, 1.7320508, 2, 2.236068},
		},
		{
			13,
			[]float32{1, 1, 1, 1},
			[]int{1, 4},
			[]float32{1, 1, 1, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		sqrt := sqrtVersions[test.version]()
		res, err := sqrt.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationSqrt(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, ops.NewBaseOperator(13, 1, 1, sqrtTypeConstraints, "sqrt")),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", ops.NewBaseOperator(13, 1, 1, sqrtTypeConstraints, "sqrt")),
		},
	}

	for _, test := range tests {
		sqrt := sqrtVersions[test.version]()
		validated, err := sqrt.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		assert.Equal(t, test.inputs, validated)
	}
}
