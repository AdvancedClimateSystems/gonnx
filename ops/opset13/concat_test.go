package opset13

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
	err := concat.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{}})

	expected := ops.ErrInvalidAttributeCount(1, 0, concat)
	assert.Equal(t, expected, err)
}

func TestConcat(t *testing.T) {
	tests := []struct {
		concat          *Concat
		backings        [][]float32
		shapes          [][]int
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			&Concat{1, 2, [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}},
			[][]float32{{0, 1, 2, 3}, {10, 20}},
			[][]int{{2, 2}, {2, 1}},
			[]int{2, 3},
			[]float32{0, 1, 10, 2, 3, 20},
		},
		{
			&Concat{1, 2, [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}},
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

		res, err := test.concat.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationConcat(t *testing.T) {
	tests := []struct {
		concat ops.Operator
		inputs []tensor.Tensor
	}{
		{
			&Concat{1, 2, [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
		},
		{
			&Concat{1, 1, [][]tensor.Dtype{ops.AllTypes}},
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
		},
	}

	for _, test := range tests {
		validated, err := test.concat.ValidateInputs(test.inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.inputs, validated)
	}
}
