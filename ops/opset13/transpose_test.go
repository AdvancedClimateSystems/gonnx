package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTransposeInit(t *testing.T) {
	trans := &Transpose{}
	err := trans.Init(TransposeOnnxAttributeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []int{1, 0}, trans.perm)
}

func TestTransposeInitFailWrongAttribute(t *testing.T) {
	trans := &Transpose{}
	err := trans.Init([]*onnx.AttributeProto{{Name: "unknownAttribute"}})

	expected := fmt.Errorf(ops.UnknownAttributeErrTemplate, trans, "unknownAttribute")
	assert.Equal(t, expected, err)
}

func TestTransposeInitFailAttrCount(t *testing.T) {
	trans := &Transpose{}
	err := trans.Init([]*onnx.AttributeProto{})

	expected := fmt.Errorf(ops.InvalidAttrCountErrTemplate, trans, 1, 0)
	assert.Equal(t, expected, err)
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		trans           *Transpose
		shape           []int
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			&Transpose{},
			[]int{3, 2},
			[]int{2, 3},
			[]float32{0, 2, 4, 1, 3, 5},
		},
		{
			&Transpose{perm: []int{0, 2, 1}},
			[]int{1, 2, 3},
			[]int{1, 3, 2},
			[]float32{0, 3, 1, 4, 2, 5},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.shape...),
		}
		res, err := test.trans.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationTranspose(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("transpose operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("transpose operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		transpose := &Transpose{}
		validated, err := transpose.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func TransposeOnnxAttributeProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{
		{Name: "perm", Ints: []int64{1, 0}},
	}
}
