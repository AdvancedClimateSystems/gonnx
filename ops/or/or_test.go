package or

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestOrInit(t *testing.T) {
	o := &Or{}

	// since 'or' does not have any attributes we pass in nil. This should not
	// fail initializing the or.
	err := o.Init(nil)
	assert.Nil(t, err)
}

func TestOr(t *testing.T) {
	tests := []struct {
		version  int64
		backings [][]bool
		shapes   [][]int
		expected []bool
	}{
		{
			7,
			[][]bool{{true, false, true, false}, {true, true, true, false}},
			[][]int{{2, 2}, {2, 2}},
			[]bool{true, true, true, false},
		},
		{
			7,
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {1, 2}},
			[]bool{true, false, true, false},
		},
		{
			7,
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {2, 1}},
			[]bool{true, true, true, false},
		},
		{
			7,
			[][]bool{{true, false, true, false, true, false}, {false, false}},
			[][]int{{3, 2}, {1, 2}},
			[]bool{true, false, true, false, true, false},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		or := orVersions[test.version]()
		res, err := or.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationOr(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			ops.ErrInvalidInputCount(1, or7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(1, "int", or7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		or := orVersions[test.version]()
		validated, err := or.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func or7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(
		7,
		2,
		2,
		orTypeConstraints,
		"or",
	)
}
