package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGreaterOrEqualInit(t *testing.T) {
	g := &GreaterOrEqual{}

	// since 'greaterOrEqual' does not have any attributes we pass in nil. This should not
	// fail initializing the greaterOrEqual.
	err := g.Init(nil)
	assert.Nil(t, err)
}

func TestGreaterOrEqual(t *testing.T) {
	tests := []struct {
		greaterOrEqual *GreaterOrEqual
		backings       [][]float32
		shapes         [][]int
		expected       []bool
	}{
		{
			&GreaterOrEqual{},
			[][]float32{{0, 1, 2, 3}, {1, 1, 1, 1}},
			[][]int{{2, 2}, {2, 2}},
			[]bool{false, true, true, true},
		},
		{
			&GreaterOrEqual{},
			[][]float32{{0, 1, 2, 3, 4, 5}, {2, 2, 2, 2, 2, 2}},
			[][]int{{3, 2}, {3, 2}},
			[]bool{false, false, true, true, true, true},
		},
		{
			&GreaterOrEqual{},
			[][]float32{{0, 1}, {0, 1, 2, 3}},
			[][]int{{2}, {2, 2}},
			[]bool{true, true, false, false},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		res, err := test.greaterOrEqual.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationGreaterOrEqual(t *testing.T) {
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
			ops.ErrInvalidInputCount(1, &GreaterOrEqual{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &GreaterOrEqual{}),
		},
	}

	for _, test := range tests {
		greaterOrEqual := &GreaterOrEqual{}
		validated, err := greaterOrEqual.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
