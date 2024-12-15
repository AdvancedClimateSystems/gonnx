package cosh

import (
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
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			9,
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
			[]float32{3.7621956, 1.5430807, 1, 1.5430807},
		},
		{
			9,
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{1.5430807, 10.067662, 27.308233, 74.209946},
		},
		{
			9,
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{1.5430807, 1.5430807, 1.5430807, 1.5430807},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		cosh := coshVersions[test.version]()

		res, err := cosh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationCosh(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, cosh9BaseOpFixture()),
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", cosh9BaseOpFixture()),
		},
	}

	for _, test := range tests {
		cosh := coshVersions[test.version]()
		validated, err := cosh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func cosh9BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(9, 1, 1, coshTypeConstraints, "cosh")
}
