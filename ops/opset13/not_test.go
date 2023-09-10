package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestNotInit(t *testing.T) {
	n := &Not{}

	// since 'not' does not have any attributes we pass in nil. This should not
	// fail initializing the not.
	err := n.Init(nil)
	assert.Nil(t, err)
}

func TestNot(t *testing.T) {
	tests := []struct {
		not      *Not
		backing  []bool
		shape    []int
		expected []bool
	}{
		{
			&Not{},
			[]bool{true, false, true, false},
			[]int{2, 2},
			[]bool{false, true, false, true},
		},
		{
			&Not{},
			[]bool{true, true, false, false},
			[]int{1, 4},
			[]bool{false, false, true, true},
		},
		{
			&Not{},
			[]bool{false, false, false, false},
			[]int{4, 1},
			[]bool{true, true, true, true},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.not.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationNot(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("not operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("not operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		not := &Not{}
		validated, err := not.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
