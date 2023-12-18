package opset13

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
		or       *Or
		backings [][]bool
		shapes   [][]int
		expected []bool
	}{
		{
			&Or{},
			[][]bool{{true, false, true, false}, {true, true, true, false}},
			[][]int{{2, 2}, {2, 2}},
			[]bool{true, true, true, false},
		},
		{
			&Or{},
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {1, 2}},
			[]bool{true, false, true, false},
		},
		{
			&Or{},
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {2, 1}},
			[]bool{true, true, true, false},
		},
		{
			&Or{},
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

		res, err := test.or.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationOr(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			ops.ErrInvalidInputCount(1, &Or{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(1, "int", &Or{}),
		},
	}

	for _, test := range tests {
		or := &Or{}
		validated, err := or.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
