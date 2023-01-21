package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSigmoidInit(t *testing.T) {
	s := newSigmoid()
	// Since the sigmoid does not have any attributes we expect it to initialize even
	// when nil is passed.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			[]float32{-4, -3, -2, -1, 0, 12},
			[]int{3, 2},
			[]float32{
				0.01798620996209155802679,
				0.04742587317756678087885, 0.1192029220221175559403,
				0.2689414213699951207488, 0.5,
				0.9999938558253977852822,
			},
		},
		{
			[]float32{-4, -4, -4, 3, 2, 1},
			[]int{3, 2},
			[]float32{
				0.01798621, 0.01798621, 0.01798621,
				0.95257413, 0.8807971, 0.7310586,
			},
		},
		{
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{4, 3},
			[]float32{
				0.5, 0.7310586, 0.8807971, 0.95257413,
				0.98201376, 0.9933072, 0.99752736, 0.99908894,
				0.99966466, 0.9998766, 0.9999546, 0.9999833,
			},
		},
	}

	for _, test := range tests {
		sigmoid := &Sigmoid{}
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
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("sigmoid operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("sigmoid operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		sigmoid := &Sigmoid{}
		validated, err := sigmoid.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
