package opset13

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"
	"gorgonia.org/tensor"
)

func TestSubInit(t *testing.T) {
	sub := &Sub{}

	// since the sub does not have any attributes we pass in nil. This should not
	// fail initializing the sub.
	err := sub.Init(nil)
	assert.Nil(t, err)
}

func TestSub(t *testing.T) {
	tests := []struct {
		shapes   [][]int
		backings [][]float32
		expected []float32
	}{
		{
			[][]int{{2, 2}, {2, 2}},
			[][]float32{{1, 1, 1, 1}, {0, 1, 2, 3}},
			[]float32{1, 0, -1, -2},
		},
		{
			[][]int{{2, 2}, {2}},
			[][]float32{{1, 1, 1, 1}, {0, 1}},
			[]float32{1, 0, 1, 0},
		},
		{
			[][]int{{2, 2}, {1}},
			[][]float32{{1, 1, 1, 1}, {2}},
			[]float32{-1, -1, -1, -1},
		},
	}

	for _, test := range tests {
		sub := &Sub{}
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}
		res, err := sub.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationSub(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("sub operator: expected 2 input tensors, got 1"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			fmt.Errorf("sub operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		sub := &Sub{}
		validated, err := sub.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
