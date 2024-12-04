package sigmoid

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSigmoidInit(t *testing.T) {
	s := &Sigmoid{}
	// Since the sigmoid does not have any attributes we expect it to initialize even
	// when nil is passed.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			6,
			[]float32{-2, -1, 0, 3},
			[]int{2, 2},
			[]float32{
				0.11920292,
				0.26894143,
				0.5,
				0.95257413,
			},
		},
		{
			13,
			[]float32{-4, -3, -2, -1, 0, 12},
			[]int{3, 2},
			[]float32{
				0.01798620996209155802679,
				0.04742587317756678087885, 0.1192029220221175559403,
				0.26894142699951207488, 0.5,
				0.9999938558253977852822,
			},
		},
		{
			13,
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{
				0.01798621, 0.01798621, 0.01798621,
				0.952574, 0.8807971, 0.7310586,
			},
		},
		{
			13,
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{4, 3},
			[]float32{
				0.5, 0.7310586, 0.8807971, 0.952574,
				0.98201376, 0.9933072, 0.99752736, 0.99908894,
				0.99966466, 0.9998766, 0.9999546, 0.9999833,
			},
		},
	}

	for _, test := range tests {
		sigmoid := sigmoidVersions[test.version]()
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := sigmoid.Apply(inputs)
		assert.Nil(t, err)
		assert.InDeltaSlice(t, test.expected, res[0].Data(), 0.00001)
	}
}

func TestInputValidationSigmoid(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
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
			ops.ErrInvalidInputCount(0, sigmoid6BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, sigmoid13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", sigmoid13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		sigmoid := sigmoidVersions[test.version]()
		validated, err := sigmoid.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func sigmoid6BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(6, 1, 1, sigmoidTypeConstraints, "sigmoid")
}

func sigmoid13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, sigmoidTypeConstraints, "sigmoid")
}
