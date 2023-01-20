package opset13

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/onnx"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"

	"gorgonia.org/tensor"
)

func TensorProtoFromNumber(n interface{}) *onnx.TensorProto {
	switch x := n.(type) {
	case int8:
		// We have to manually make the binary version for rawData
		size := 1
		rawData := make([]byte, size)
		rawData[0] = uint8(x)
		return &onnx.TensorProto{
			DataType: onnx.TensorProto_DataType_value["INT8"],
			Dims:     []int64{1},
			RawData:  rawData,
		}
	case int16:
		// We have to manually make the binary version for rawData
		size := 2
		rawData := make([]byte, size)
		binary.LittleEndian.PutUint16(rawData, uint16(x))
		return &onnx.TensorProto{
			DataType: onnx.TensorProto_DataType_value["INT16"],
			Dims:     []int64{1},
			RawData:  rawData,
		}
	case int32:
		return &onnx.TensorProto{
			DataType:  onnx.TensorProto_DataType_value["INT32"],
			Dims:      []int64{1},
			Int32Data: []int32{x},
		}
	case int64:
		return &onnx.TensorProto{
			DataType:  onnx.TensorProto_DataType_value["INT64"],
			Dims:      []int64{1},
			Int64Data: []int64{x},
		}
	case float32:
		return &onnx.TensorProto{
			DataType:  onnx.TensorProto_DataType_value["FLOAT32"],
			Dims:      []int64{1},
			FloatData: []float32{x},
		}
	case float64:
		return &onnx.TensorProto{
			DataType:   onnx.TensorProto_DataType_value["DOUBLE"],
			Dims:       []int64{1},
			DoubleData: []float64{x},
		}
	default:
		return nil
	}
}

func TestConstantOfShape(t *testing.T) {
	// Test cases, verifying that all these types work.
	// Unfortunately uint* and bool are not supported.
	tests := []struct {
		input        interface{}
		expectTensor interface{}
	}{
		{float32(42.0), []float32{42.0, 42.0, 42.0, 42.0}},
		{float64(42.0), []float64{42.0, 42.0, 42.0, 42.0}},
		{int8(42), []int8{42.0, 42.0, 42.0, 42.0}},
		{int16(42), []int16{42.0, 42.0, 42.0, 42.0}},
		{int32(42), []int32{42.0, 42.0, 42.0, 42.0}},
		{int64(42), []int64{42.0, 42.0, 42.0, 42.0}},
		{int32(-1), []int32{-1, -1, -1, -1}},
		{int32(0), []int32{0, 0, 0, 0}},
	}

	for _, test := range tests {
		testFunc := func(t *testing.T) {
			// Make the input tensor
			tp := TensorProtoFromNumber(test.input)
			assert.NotNil(t, tp)
			attr := []*onnx.AttributeProto{{Name: "value", T: tp}}

			// Create operator
			op := ConstantOfShape{}
			err := op.Init(attr)
			assert.NoError(t, err)
			assert.Equal(t, test.input, op.value.Data())

			shape := []int64{2, 2}
			input := tensor.New(tensor.WithBacking(shape))

			res, err := op.Apply([]tensor.Tensor{input})
			assert.NoError(t, err)
			assert.Equal(t, test.expectTensor, res[0].Data())
		}
		t.Run("Test ", testFunc)
	}
}

func TestConstantOfShapeEmptyInit(t *testing.T) {
	op := &ConstantOfShape{}

	// No init value given
	err := op.Init([]*onnx.AttributeProto{})
	assert.NoError(t, err)

	assert.Equal(t, float32(0.0), op.value.Data())

	shape := []int64{2, 2}

	input := tensor.New(tensor.WithBacking(shape))
	res, err := op.Apply([]tensor.Tensor{input})
	assert.NoError(t, err)

	assert.Equal(t, []float32{0, 0, 0, 0}, res[0].Data())

}

func TestIncorrectInput(t *testing.T) {
	tp := &onnx.TensorProto{
		DataType:  onnx.TensorProto_DataType_value["INT32"],
		Dims:      []int64{3},
		Int32Data: []int32{1, 2, 3},
	}
	attr := []*onnx.AttributeProto{{Name: "value", T: tp}}

	op := &ConstantOfShape{}
	err := op.Init(attr)
	assert.NotNil(t, err)
	assert.Equal(
		t,
		"Value input tensor should be a single element tensor, but was [1  2  3]",
		err.Error(),
	)

}

func TestNegativeShapeNotAllowed(t *testing.T) {
	op := &ConstantOfShape{}
	op.Init([]*onnx.AttributeProto{})

	shape := []int64{1, -1}

	input := tensor.New(tensor.WithBacking(shape))
	_, err := op.Apply([]tensor.Tensor{input})
	assert.NotNil(t, err)

	assert.Equal(
		t,
		"Non positive dimensions are not allowed (must be > 0). Given: [1 -1]",
		err.Error())
}

func TestEmptyTensorNotAllowed(t *testing.T) {
	op := &ConstantOfShape{}
	op.Init([]*onnx.AttributeProto{})

	shape := []int64{0}

	input := tensor.New(tensor.WithBacking(shape))
	_, err := op.Apply([]tensor.Tensor{input})
	assert.NotNil(t, err)

	assert.Equal(
		t,
		"Non positive dimensions are not allowed (must be > 0). Given: [0]",
		err.Error())
}

func TestScalarShapeInput(t *testing.T) {
	op := &ConstantOfShape{}
	op.Init([]*onnx.AttributeProto{})

	shape := []int64{6}
	input := tensor.New(tensor.WithBacking(shape))

	res, err := op.Apply([]tensor.Tensor{input})

	assert.NoError(t, err)
	assert.Equal(t, []float32{0, 0, 0, 0, 0, 0}, res[0].Data())
}

func TestInputValidationConstantOfShape(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1}, 1),
			},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("constant of shape operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("constant of shape operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		constantOfShape := &ConstantOfShape{}
		validated, err := constantOfShape.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
