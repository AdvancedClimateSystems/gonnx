package greater

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGreaterInit(t *testing.T) {
	g := &Greater{}

	// since 'greater' does not have any attributes we pass in nil. This should not
	// fail initializing the greater.
	err := g.Init(ops.EmptyNodeProto())
	assert.Nil(t, err)
}

func TestGreater(t *testing.T) {
	tests := []struct {
		version  int64
		backings [][]float32
		shapes   [][]int
		expected []bool
	}{
		{
			7,
			[][]float32{{0, 1, 2, 3}, {1, 1, 1, 1}},
			[][]int{{2, 2}, {2, 2}},
			[]bool{false, false, true, true},
		},
		{
			9,
			[][]float32{{0, 1, 2, 3, 4, 5}, {2, 2, 2, 2, 2, 2}},
			[][]int{{3, 2}, {3, 2}},
			[]bool{false, false, false, true, true, true},
		},
		{
			13,
			[][]float32{{0, 1}, {0, 1, 2, 3}},
			[][]int{{2}, {2, 2}},
			[]bool{false, false, false, false},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		greater := greaterVersions[test.version]()

		res, err := greater.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationGreater(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint64{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, greater13BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int32", greater7BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int", greater13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		greater := greaterVersions[test.version]()
		validated, err := greater.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func greater7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 2, 2, greater7TypeConstraints, "greater")
}

func greater13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 2, greaterTypeConstraints, "greater")
}
