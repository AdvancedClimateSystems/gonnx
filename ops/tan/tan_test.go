package tan

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTanInit(t *testing.T) {
	a := &Tan{}

	// since 'tan' does not have any attributes we pass in nil. This should not
	// fail initializing the tan.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestTan(t *testing.T) {
	tests := []struct {
		tan      *Tan
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Tan{},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{1.5574077, -2.1850398, -0.14254655, 1.1578213},
		},
		{
			&Tan{},
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{1.5574077, -2.1850398, -0.14254655, 1.1578213},
		},
		{
			&Tan{},
			[]float32{2, 2, 2, 2},
			[]int{1, 4},
			[]float32{-2.1850398, -2.1850398, -2.1850398, -2.1850398},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.tan.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationTan(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, tan7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", tan7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		tan := tanVersions[test.version]()
		validated, err := tan.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func tan7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 1, 1, tanTypeConstraints, "tan")
}
