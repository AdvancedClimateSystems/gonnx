package concat

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConcatInit(t *testing.T) {
	concat := &Concat{}
	err := concat.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 3}}})

	assert.Nil(t, err)
	assert.Equal(t, 3, concat.axis)
}

func TestConcatInitFail(t *testing.T) {
	concat := &Concat{}
	err := concat.Init(ops.EmptyNodeProto())

	expected := ops.ErrInvalidAttributeCount(1, 0, concat)
	assert.Equal(t, expected, err)
}

func TestConcat(t *testing.T) {
	tests := []struct {
		version         int64
		node            *onnx.NodeProto
		backings        [][]float32
		shapes          [][]int
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			13,
			&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 1}}},
			[][]float32{{0, 1, 2, 3}, {10, 20}},
			[][]int{{2, 2}, {2, 1}},
			[]int{2, 3},
			[]float32{0, 1, 10, 2, 3, 20},
		},
		{
			13,
			&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 1}}},
			[][]float32{{0, 1, 2, 3}, {10, 20, 30, 40, 50, 60}},
			[][]int{{2, 2}, {2, 3}},
			[]int{2, 5},
			[]float32{0, 1, 10, 20, 30, 2, 3, 40, 50, 60},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		concat := concatVersions[test.version]()
		err := concat.Init(test.node)
		assert.Nil(t, err)

		inputs, err = concat.ValidateInputs(inputs)
		assert.Nil(t, err)

		res, err := concat.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationConcat(t *testing.T) {
	tests := []struct {
		version int64
		node    *onnx.NodeProto
		inputs  []tensor.Tensor
	}{
		{
			13,
			&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 1}}},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
		},
		{
			13,
			&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 1}}},
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
		},
	}

	for _, test := range tests {
		concat := concatVersions[test.version]()
		err := concat.Init(test.node)
		assert.Nil(t, err)

		validated, err := concat.ValidateInputs(test.inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.inputs, validated)
	}
}
