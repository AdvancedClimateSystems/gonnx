package opset13

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"
	"gorgonia.org/tensor"
)

func TestReshapeInit(t *testing.T) {
	r := &Reshape{}

	// since the reshape does not have any attributes we pass in nil. This should not
	// fail initializing the reshape.
	err := r.Init(nil)
	assert.Nil(t, err)
}

func TestReshape(t *testing.T) {
	tests := []struct {
		inputShape []int
		newShape   []int64
		expected   tensor.Shape
	}{
		{
			[]int{2, 3},
			[]int64{1, 6},
			[]int{1, 6},
		},
		{
			[]int{1, 2, 3},
			[]int64{0, 2, 3},
			[]int{1, 2, 3},
		},
		{
			[]int{1, 2, 3},
			[]int64{1, -1, 2},
			[]int{1, 3, 2},
		},
		{
			[]int{1, 2, 3},
			[]int64{1, -1},
			[]int{1, 6},
		},
		{
			[]int{3, 4, 2},
			[]int64{1, 0, -1},
			[]int{1, 4, 6},
		},
	}

	for _, test := range tests {
		reshape := &Reshape{}
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.inputShape...),
			tensor.New(tensor.WithBacking(test.newShape)),
		}
		res, err := reshape.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Shape())
	}
}

func TestInputValidationReshape(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("reshape operator: expected 2 input tensors, got 1"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			fmt.Errorf("reshape operator: input 1 does not allow type int"),
		},
	}

	for _, test := range tests {
		reshape := &Reshape{}
		validated, err := reshape.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
