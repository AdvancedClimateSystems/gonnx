package xor

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestXorInit(t *testing.T) {
	x := &Xor{}

	err := x.Init(nil)
	assert.Nil(t, err)
}

func TestXor(t *testing.T) {
	tests := []struct {
		xor      *Xor
		backings [][]bool
		shapes   [][]int
		expected []bool
	}{
		{
			&Xor{},
			[][]bool{{true, false, true, false}, {true, true, true, false}},
			[][]int{{2, 2}, {2, 2}},
			[]bool{false, true, false, false},
		},
		{
			&Xor{},
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {1, 2}},
			[]bool{false, false, false, false},
		},
		{
			&Xor{},
			[][]bool{{true, false, true, false}, {true, false}},
			[][]int{{2, 2}, {2, 1}},
			[]bool{false, true, true, false},
		},
		{
			&Xor{},
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

		res, err := test.xor.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationXor(t *testing.T) {
	tests := []struct {
		inputs  []tensor.Tensor
		err     error
		version int64
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			nil,
			7,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
			},
			ops.ErrInvalidInputCount(1, ops.NewBaseOperator(7, 2, 2, xorTypeConstraints, "xor")),
			7,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{false, false}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(1, "int", ops.NewBaseOperator(7, 2, 2, xorTypeConstraints, "xor")),
			7,
		},
	}

	for _, test := range tests {
		xor := xorVersions[test.version]()
		validated, err := xor.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
