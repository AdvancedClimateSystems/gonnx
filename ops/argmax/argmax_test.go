package argmax

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestArgMaxInit(t *testing.T) {
	a := &ArgMax{}

	err := a.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "axis", I: 2},
				{Name: "keepdims", I: 0},
				{Name: "select_last_index", I: 0},
			},
		},
	)
	assert.Nil(t, err)

	assert.Equal(t, 2, a.axis)
	assert.Equal(t, false, a.keepDims)
	assert.Equal(t, false, a.selectLastIndex)
}

func TestArgMax(t *testing.T) {
	tests := []struct {
		version       int64
		node          *onnx.NodeProto
		backing       []float32
		shape         []int
		expectedShape tensor.Shape
		expectedData  []int64
	}{
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: 0},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int{1, 2},
			[]int64{1, 1},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: -1},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int{2, 1},
			[]int64{1, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		argmax := argMaxVersions[test.version]()
		err := argmax.Init(test.node)
		assert.Nil(t, err)

		res, err := argmax.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedData, res[0].Data())
	}
}

func TestInputValidationArgMax(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
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
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, argMax13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", argMax13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		argmax := argMaxVersions[test.version]()
		validated, err := argmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func argMax13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, argMaxTypeConstraints, "argmax")
}
