package tanh

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTanhInit(t *testing.T) {
	tanh := &Tanh{}
	// Since the tanh does not have any attributes we expect it to initialize even
	// when nil is passed.
	err := tanh.Init(nil)

	assert.Nil(t, err)
}

func TestTanh(t *testing.T) {
	tests := []struct {
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			[]float32{-4, -3, -2, -1, 0, 12},
			[]int{3, 2},
			[]float32{-0.9993293, -0.9950548, -0.9640276, -0.7615942, 0, 1},
		},
		{
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{-0.9993293, -0.9993293, -0.9993293, 0.9950548, 0.9640276, 0.7615942},
		},
		{
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{4, 3},
			[]float32{
				0, 0.7615942, 0.9640276, 0.9950548, 0.9993293, 0.9999092,
				0.9999877, 0.99999833, 0.99999976, 0.99999994, 1, 1,
			},
		},
	}

	for _, test := range tests {
		tanh := &Tanh{}
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := tanh.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationTanh(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			6,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			6,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, tanh6BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, tanh13BaseOpFixture()),
		},
		{
			6,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", tanh6BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", tanh13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		tanh := tanhVersions[test.version]()
		validated, err := tanh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func tanh6BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(6, 1, 1, tanhTypeConstraint, "tanh")
}

func tanh13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, tanhTypeConstraint, "tanh")
}
