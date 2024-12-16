package constant

import (
	"encoding/binary"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConstantInit(t *testing.T) {
	constant := Constant{}

	tests := []struct {
		version  int64
		initAttr []*onnx.AttributeProto
		expected interface{}
		err      error
	}{
		{
			13,
			ConstantValueAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 1, 1})),
			nil,
		},
		{
			13,
			ConstantValueFloatAttrProtoFixture(),
			tensor.New(tensor.FromScalar(float32(0.2))),
			nil,
		},
		{
			13,
			ConstantValueFloatsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]float32{0.1, 0.2})),
			nil,
		},
		{
			13,
			ConstantValueIntAttrProtoFixture(),
			tensor.New(tensor.FromScalar(int64(1))),
			nil,
		},
		{
			13,
			ConstantValueIntsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 2, 3})),
			nil,
		},
		{
			13,
			[]*onnx.AttributeProto{{Name: "sparse_value"}},
			nil,
			ops.ErrUnsupportedAttribute("sparse_value", &constant),
		},
		{
			13,
			[]*onnx.AttributeProto{{Name: "unknownAttribute"}},
			nil,
			ops.ErrUnsupportedAttribute("unknownAttribute", &constant),
		},
		{
			13,
			[]*onnx.AttributeProto{},
			nil,
			ops.ErrInvalidAttributeCount(1, 0, &constant),
		},
	}

	for _, test := range tests {
		constant.value = nil
		err := constant.Init(&onnx.NodeProto{Attribute: test.initAttr})

		assert.Equal(t, test.err, err)

		if err != nil {
			assert.Equal(t, test.expected, constant.value)
		}
	}
}

func TestConstant(t *testing.T) {
	tests := []struct {
		version  int64
		initAttr []*onnx.AttributeProto
		expected interface{}
	}{
		{
			13,
			ConstantValueAttrProtoFixture(),
			[]int64{1, 1, 1},
		},
		{
			13,
			ConstantValueFloatAttrProtoFixture(),
			float32(0.2),
		},
		{
			13,
			ConstantValueFloatsAttrProtoFixture(),
			[]float32{0.1, 0.2},
		},
		{
			13,
			ConstantValueIntAttrProtoFixture(),
			int64(1),
		},
		{
			13,
			ConstantValueIntsAttrProtoFixture(),
			[]int64{1, 2, 3},
		},
	}

	for _, test := range tests {
		constant := constantVersions[test.version]()
		_ = constant.Init(&onnx.NodeProto{Attribute: test.initAttr})
		res, err := constant.Apply([]tensor.Tensor{})
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestConstantSingleIntShapeTensor(t *testing.T) {
	constant := &Constant{}
	err := constant.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "value_ints", Ints: []int64{2}}}})

	assert.Nil(t, err)
	assert.False(t, constant.value.IsScalar())
}

func TestInputValidationConstant(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, constant13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		constant := constantVersions[test.version]()
		validated, err := constant.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func ConstantValueAttrProtoFixture() []*onnx.AttributeProto {
	values := []int64{1, 1, 1}
	bValues := make([]byte, 24)

	binary.LittleEndian.PutUint64(bValues[:8], uint64(values[0]))
	binary.LittleEndian.PutUint64(bValues[8:16], uint64(values[1]))
	binary.LittleEndian.PutUint64(bValues[16:24], uint64(values[2]))

	tp := &onnx.TensorProto{DataType: int32(7), Dims: []int64{3}, RawData: bValues}

	return []*onnx.AttributeProto{{Name: "value", T: tp}}
}

func ConstantValueFloatAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_float", F: float32(0.2)}}
}

func ConstantValueFloatsAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_floats", Floats: []float32{0.1, 0.2}}}
}

func ConstantValueIntAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_int", I: int64(1)}}
}

func ConstantValueIntsAttrProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{{Name: "value_ints", Ints: []int64{1, 2, 3}}}
}

func constant13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 0, 0, [][]tensor.Dtype{}, "constant")
}
