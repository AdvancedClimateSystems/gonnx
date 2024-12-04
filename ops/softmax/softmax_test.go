package softmax

import (
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
		// This axis is out of range, because the input tensor only has 2 dimensions.
		axis: 3,
	}
	_, err := softmax.Apply(inputs)
	assert.Equal(
		t,
		err,
		ops.ErrAxisOutOfRange(-2, 2, 3),
	)
}

func TestInputValidationSoftmax(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			11,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
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
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, softmax1BaseOpFixture()),
		},
		{
			11,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, softmax11BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, softmax13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", softmax13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		softmax := softmaxVersions[test.version]()
		validated, err := softmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func softmax1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 1, 1, softmaxTypeConstraints, "softmax")
}

func softmax11BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(11, 1, 1, softmaxTypeConstraints, "softmax")
}

func softmax13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, softmaxTypeConstraints, "softmax")
}
