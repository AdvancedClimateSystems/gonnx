package constant

import (
	"encoding/binary"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConstant13Init(t *testing.T) {
	tests := []struct {
		initAttr []*onnx.AttributeProto
		expected interface{}
		err      error
	}{
		{
			Constant13ValueAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 1, 1})),
			nil,
		},
		{
			Constant13ValueFloatAttrProtoFixture(),
			tensor.New(tensor.FromScalar(float32(0.2))),
			nil,
		},
		{
			Constant13ValueFloatsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]float32{0.1, 0.2})),
			nil,
		},
		{
			Constant13ValueIntAttrProtoFixture(),
			tensor.New(tensor.FromScalar(int64(1))),
			nil,
		},
		{
			Constant13ValueIntsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 2, 3})),
			nil,
		},
		{
			[]*onnx.AttributeProto{{Name: "sparse_value"}},
			nil,
			ops.ErrUnsupportedAttribute("sparse_value", &Constant13{}),
		},
		{
			[]*onnx.AttributeProto{{Name: "unknownAttribute"}},
			nil,
			ops.ErrUnsupportedAttribute("unknownAttribute", &Constant13{}),
		},
		{
			[]*onnx.AttributeProto{},
			nil,
			ops.ErrInvalidAttributeCount(1, 0, &Constant13{}),
		},
	}

	for _, test := range tests {
		constant := &Constant13{}
		err := constant.Init(&onnx.NodeProto{Attribute: test.initAttr})

		assert.Equal(t, test.err, err)

		if err != nil {
			assert.Equal(t, test.expected, constant.value)
		}
	}
}

func TestConstant13(t *testing.T) {
	tests := []struct {
		constant *Constant13
		initAttr []*onnx.AttributeProto
		expected interface{}
	}{
		{
			&Constant13{},
			Constant13ValueAttrProtoFixture(),
			[]int64{1, 1, 1},
		},
		{
			&Constant13{},
			Constant13ValueFloatAttrProtoFixture(),
			float32(0.2),
		},
		{
			&Constant13{},
			Constant13ValueFloatsAttrProtoFixture(),
			[]float32{0.1, 0.2},
		},
		{
			&Constant13{},
			Constant13ValueIntAttrProtoFixture(),
			int64(1),
		},
		{
			&Constant13{},
			Constant13ValueIntsAttrProtoFixture(),
			[]int64{1, 2, 3},
		},
	}

	for _, test := range tests {
		_ = test.constant.Init(&onnx.NodeProto{Attribute: test.initAttr})
		res, err := test.constant.Apply([]tensor.Tensor{})
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestConstant13SingleIntShapeTensor(t *testing.T) {
	constant := &Constant13{}
	err := constant.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "value_ints", Ints: []int64{2}}}})

	assert.Nil(t, err)
	assert.False(t, constant.value.IsScalar())
}

func TestInputValidationConstant13(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, &Constant13{}),
		},
	}

	for _, test := range tests {
		constant := &Constant13{}
		validated, err := constant.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func Constant13ValueAttrProtoFixture() []*onnx.AttributeProto {
	values := []int64{1, 1, 1}
	bValues := make([]byte, 24)

	binary.LittleEndian.PutUint64(bValues[:8], uint64(values[0]))
	binary.LittleEndian.PutUint64(bValues[8:16], uint64(values[1]))
	binary.LittleEndian.PutUint64(bValues[16:24], uint64(values[2]))

	tp := &onnx.TensorProto{DataType: int32(7), Dims: []int64{3}, RawData: bValues}

	return []*onnx.AttributeProto{{Name: "value", T: tp}}
}

func Constant13ValueFloatAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_float", F: float32(0.2)}}
}

func Constant13ValueFloatsAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_floats", Floats: []float32{0.1, 0.2}}}
}

func Constant13ValueIntAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_int", I: int64(1)}}
}

func Constant13ValueIntsAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_ints", Ints: []int64{1, 2, 3}}}
}
