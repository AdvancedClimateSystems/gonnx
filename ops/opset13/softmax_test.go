package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSoftmaxInit(t *testing.T) {
	s := &Softmax{}

	// since 'softmax' does not have any attributes we pass in nil. This should not
	// fail initializing the softmax.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		softmax  *Softmax
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Softmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{0.032058604, 0.087144315, 0.2368828, 0.6439142},
		},
		{
			&Softmax{
				axis: 1,
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{0.032058604, 0.087144315, 0.2368828, 0.6439142},
		},
		{
			&Softmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0.26894143, 0.7310586, 0.26894143, 0.7310586},
		},
		{
			&Softmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{1, 2, 3},
			[]float32{0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094},
		},
		{
			&Softmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{4, 1},
			[]float32{1, 1, 1, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.softmax.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestSoftmaxFail(t *testing.T) {
	inputs := []tensor.Tensor{
		ops.TensorWithBackingFixture([]float32{1, 2, 3, 4}, 2, 2),
	}

	softmax := &Softmax{
		axis: 3, // This axis is out of range.
	}
	_, err := softmax.Apply(inputs)
	assert.Equal(
		t,
		err,
		fmt.Errorf(
			ops.AxisOutOfRangeErrTemplate,
			2,
			2,
			3,
		),
	)
}

func TestInputValidationSoftmax(t *testing.T) {
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
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("softmax operator: expected 1 input tensors, got 2"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("softmax operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		softmax := &Softmax{axis: -1}
		validated, err := softmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
