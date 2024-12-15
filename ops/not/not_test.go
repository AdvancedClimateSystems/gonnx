package not

import (
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
		version  int64
		backing  []bool
		shape    []int
		expected []bool
	}{
		{
			1,
			[]bool{true, false, true, false},
			[]int{2, 2},
			[]bool{false, true, false, true},
		},
		{
			1,
			[]bool{true, true, false, false},
			[]int{1, 4},
			[]bool{false, false, true, true},
		},
		{
			1,
			[]bool{false, false, false, false},
			[]int{4, 1},
			[]bool{true, true, true, true},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		not := notVersions[test.version]()
		res, err := not.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationNot(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			nil,
		},
		{
			1,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, not1BaseOpFixture()),
		},
		{
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", not1BaseOpFixture()),
		},
	}

	for _, test := range tests {
		not := notVersions[test.version]()
		validated, err := not.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func not1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(
		1,
		1,
		1,
		notTypeConstraints,
		"not",
	)
}
