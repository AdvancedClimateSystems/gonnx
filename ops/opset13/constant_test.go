package opset13

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConstantInit(t *testing.T) {
	tests := []struct {
		initAttr []*onnx.AttributeProto
		expected interface{}
		err      error
	}{
		{
			ConstantValueAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 1, 1})),
			nil,
		},
		{
			ConstantValueFloatAttrProtoFixture(),
			tensor.New(tensor.FromScalar(float32(0.2))),
			nil,
		},
		{
			ConstantValueFloatsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]float32{0.1, 0.2})),
			nil,
		},
		{
			ConstantValueIntAttrProtoFixture(),
			tensor.New(tensor.FromScalar(int64(1))),
			nil,
		},
		{
			ConstantValueIntsAttrProtoFixture(),
			tensor.New(tensor.WithBacking([]int64{1, 2, 3})),
			nil,
		},
		{
			[]*onnx.AttributeProto{{Name: "sparse_value"}},
			nil,
			fmt.Errorf(ops.UnsupportedAttrErrTemplate, &Constant{}, "sparse_value"),
                        ops.ErrUnsupportedAttribute("sparse_value", &Constant{})
		},
		{
			[]*onnx.AttributeProto{{Name: "unknownAttribute"}},
			nil,
			fmt.Errorf(ops.UnknownAttributeErrTemplate, &Constant{}, "unknownAttribute"),
		},
		{
			[]*onnx.AttributeProto{},
			nil,
			fmt.Errorf(ops.InvalidAttrCountErrTemplate, &Constant{}, 1, 0),
		},
	}

	for _, test := range tests {
		constant := &Constant{}
		err := constant.Init(test.initAttr)

		assert.Equal(t, test.err, err)
		if err != nil {
			assert.Equal(t, test.expected, constant.value)
		}
	}
}

func TestConstant(t *testing.T) {
	tests := []struct {
		constant *Constant
		initAttr []*onnx.AttributeProto
		expected interface{}
	}{
		{
			&Constant{},
			ConstantValueAttrProtoFixture(),
			[]int64{1, 1, 1},
		},
		{
			&Constant{},
			ConstantValueFloatAttrProtoFixture(),
			float32(0.2),
		},
		{
			&Constant{},
			ConstantValueFloatsAttrProtoFixture(),
			[]float32{0.1, 0.2},
		},
		{
			&Constant{},
			ConstantValueIntAttrProtoFixture(),
			int64(1),
		},
		{
			&Constant{},
			ConstantValueIntsAttrProtoFixture(),
			[]int64{1, 2, 3},
		},
	}

	for _, test := range tests {
		test.constant.Init(test.initAttr)
		res, err := test.constant.Apply([]tensor.Tensor{})
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestConstantSingleIntShapeTensor(t *testing.T) {
	constant := &Constant{}
	err := constant.Init([]*onnx.AttributeProto{{Name: "value_ints", Ints: []int64{2}}})

	assert.Nil(t, err)
	assert.False(t, constant.value.IsScalar())
}

func TestInputValidationConstant(t *testing.T) {
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
			fmt.Errorf("constant operator: expected 0 input tensors, got 1"),
		},
	}

	for _, test := range tests {
		constant := &Constant{}
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
