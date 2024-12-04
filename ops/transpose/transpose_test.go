package transpose

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTransposeInit(t *testing.T) {
	trans := &Transpose{}
	err := trans.Init(TransposeOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []int{1, 0}, trans.perm)
}

func TestTransposeInitFailWrongAttribute(t *testing.T) {
	trans := &Transpose{}
	err := trans.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "unknownAttribute"}}})

	expected := ops.ErrInvalidAttribute("unknownAttribute", trans)
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
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, transpose1BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, transpose13BaseOpFixture()),
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", transpose1BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", transpose13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		transpose := transposeVersions[test.version]()
		validated, err := transpose.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func TransposeOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "perm", Ints: []int64{1, 0}},
		},
	}
}

func transpose1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 1, 1, transposeTypeConstraint, "transpose")
}

func transpose13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, transposeTypeConstraint, "transpose")
}
