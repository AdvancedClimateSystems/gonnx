package scaler

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestScaler1Init(t *testing.T) {
	scaler := &Scaler1{}
	err := scaler.Init(Scaler1OnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.5, 2.5, 3.5}, scaler.offset.Data())
	assert.Equal(t, []float32{0.5, 1.0, 2.0}, scaler.scale.Data())
}

func TestScaler1InitFailWrongAttribute(t *testing.T) {
	scaler := &Scaler1{}
	err := scaler.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "unknownAttribute"}, {Name: "Another"}}})

	expected := ops.ErrInvalidAttribute("unknownAttribute", scaler)
	assert.Equal(t, expected, err)
}

func TestScaler1InitFailAttrCount(t *testing.T) {
	scaler := &Scaler1{}
	err := scaler.Init(ops.EmptyNodeProto())

	expected := ops.ErrInvalidAttributeCount(2, 0, scaler)
	assert.Equal(t, expected, err)
}

func TestScaler1(t *testing.T) {
	tests := []struct {
		scaler          *Scaler1
		shape           []int
		backing         []float32
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			&Scaler1{
				offset: tensor.New(tensor.WithBacking([]float32{2.0, 25.0, 450.0})),
				scale:  tensor.New(tensor.WithBacking([]float32{0.5, 1.0, 2.0})),
			},
			[]int{2, 3},
			[]float32{1, 20, 300, 4, 50, 600},
			[]int{2, 3},
			[]float32{-0.5, -5.0, -300.0, 1.0, 25.0, 300.0},
		},
		{
			&Scaler1{
				offset: tensor.New(tensor.WithBacking([]float32{2.0})),
				scale:  tensor.New(tensor.WithBacking([]float32{0.5})),
			},
			[]int{2, 3},
			[]float32{1, 2, 3, 4, 5, 6},
			[]int{2, 3},
			[]float32{-0.5, 0.0, 0.5, 1.0, 1.5, 2.0},
		},
		{
			&Scaler1{
				offset: tensor.New(tensor.WithBacking([]float32{2.0, 1.5})),
				scale:  tensor.New(tensor.WithBacking([]float32{0.5, 2.0})),
			},
			[]int{3, 2},
			[]float32{1.0, 2.0, 2.0, 1.0, 2.5, 3.5},
			[]int{3, 2},
			[]float32{-0.5, 1.0, 0.0, -1.0, 0.25, 4.0},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}
		res, err := test.scaler.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationScaler1(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int64{1, 2}, 2)},
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
			ops.ErrInvalidInputCount(0, &Scaler1{}),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", &Scaler1{}),
		},
	}

	for _, test := range tests {
		scaler := &Scaler1{}
		validated, err := scaler.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func Scaler1OnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "offset", Floats: []float32{1.5, 2.5, 3.5}},
			{Name: "scale", Floats: []float32{0.5, 1.0, 2.0}},
		},
	}
}
