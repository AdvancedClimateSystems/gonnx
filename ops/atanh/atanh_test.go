package atanh

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAtanhInit(t *testing.T) {
	a := &Atanh{}

	// since 'atanh' does not have any attributes we pass in nil. This should not
	// fail initializing the atanh.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAtanh(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			9,
			[]float32{-0.9, -0.5, 0, 0.5},
			[]int{2, 2},
			[]float32{-1.4722193, -0.54930615, 0, 0.54930615},
		},
		{
			9,
			[]float32{-0.9, -0.5, 0, 0.5},
			[]int{1, 4},
			[]float32{-1.4722193, -0.54930615, 0, 0.54930615},
		},
		{
			9,
			[]float32{0.5, 0.5, 0.5, 0.5},
			[]int{1, 4},
			[]float32{0.54930615, 0.54930615, 0.54930615, 0.54930615},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		atanh := atanhVersions[test.version]()

		res, err := atanh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAtanh(t *testing.T) {
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
			ops.ErrInvalidInputCount(0, atanh9BaseOpFixture()),
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", atanh9BaseOpFixture()),
		},
	}

	for _, test := range tests {
		atanh := atanhVersions[test.version]()
		validated, err := atanh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func atanh9BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(9, 1, 1, atanhTypeConstraints, "atanh")
}
