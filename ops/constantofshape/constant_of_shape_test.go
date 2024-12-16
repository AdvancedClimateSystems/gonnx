package constantofshape

import (
	"encoding/binary"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"

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
		version      int64
		input        interface{}
		expectTensor interface{}
	}{
		{9, float32(42.0), []float32{42.0, 42.0, 42.0, 42.0}},
		{9, float64(42.0), []float64{42.0, 42.0, 42.0, 42.0}},
		{9, int8(42), []int8{42.0, 42.0, 42.0, 42.0}},
		{9, int16(42), []int16{42.0, 42.0, 42.0, 42.0}},
		{9, int32(42), []int32{42.0, 42.0, 42.0, 42.0}},
		{9, int64(42), []int64{42.0, 42.0, 42.0, 42.0}},
		{9, int32(-1), []int32{-1, -1, -1, -1}},
		{9, int32(0), []int32{0, 0, 0, 0}},
	}

	for _, test := range tests {
		testFunc := func(t *testing.T) {
			// Make the input tensor
			tp := TensorProtoFromNumber(test.input)
			assert.NotNil(t, tp)

			node := &onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "value", T: tp}}}
			op, ok := constantOfShapeVersions[test.version]().(*ConstantOfShape)
			assert.True(t, ok)

			err := op.Init(node)

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
	err := op.Init(ops.EmptyNodeProto())
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
	node := &onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "value", T: tp}}}

	op := constantOfShapeVersions[9]()
	err := op.Init(node)
	assert.NotNil(t, err)
	assert.Equal(
		t,
		"constantofshape v9 invalid tensor found, reason: expected tensor to have one element",
		err.Error(),
	)
}

func TestNegativeShapeNotAllowed(t *testing.T) {
	op := constantOfShapeVersions[9]()
	_ = op.Init(ops.EmptyNodeProto())

	shape := []int64{1, -1}

	input := tensor.New(tensor.WithBacking(shape))
	_, err := op.Apply([]tensor.Tensor{input})
	assert.NotNil(t, err)

	assert.Equal(
		t,
		"constantofshape v9 invalid tensor found, reason: empty dimensions are not allowed",
		err.Error())
}

func TestEmptyTensorNotAllowed(t *testing.T) {
	op := constantOfShapeVersions[9]()
	_ = op.Init(ops.EmptyNodeProto())

	shape := []int64{0}

	input := tensor.New(tensor.WithBacking(shape))
	_, err := op.Apply([]tensor.Tensor{input})
	assert.NotNil(t, err)

	assert.Equal(
		t,
		"constantofshape v9 invalid tensor found, reason: empty dimensions are not allowed",
		err.Error())
}

func TestScalarShapeInput(t *testing.T) {
	op := &ConstantOfShape{}
	_ = op.Init(ops.EmptyNodeProto())

	shape := []int64{6}
	input := tensor.New(tensor.WithBacking(shape))

	res, err := op.Apply([]tensor.Tensor{input})

	assert.NoError(t, err)
	assert.Equal(t, []float32{0, 0, 0, 0, 0, 0}, res[0].Data())
}

func TestInputValidationConstantOfShape(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1}, 1),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, constantOfShape9BaseOpFixture()),
		},
		{
			9,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", constantOfShape9BaseOpFixture()),
		},
	}

	for _, test := range tests {
		constantOfShape := constantOfShapeVersions[test.version]()
		validated, err := constantOfShape.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func constantOfShape9BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(9, 1, 1, constantOfShapeTypeConstraints, "constantofshape")
}
