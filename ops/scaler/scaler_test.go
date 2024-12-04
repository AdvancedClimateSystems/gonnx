package scaler

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestScalerInit(t *testing.T) {
	scaler := &Scaler{}
	err := scaler.Init(ScalerOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.5, 2.5, 3.5}, scaler.offset.Data())
	assert.Equal(t, []float32{0.5, 1.0, 2.0}, scaler.scale.Data())
}

func TestScalerInitFailWrongAttribute(t *testing.T) {
	scaler := &Scaler{}
	err := scaler.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "unknownAttribute"}, {Name: "Another"}}})

	expected := ops.ErrInvalidAttribute("unknownAttribute", scaler)
	assert.Equal(t, expected, err)
}

func TestScalerInitFailAttrCount(t *testing.T) {
	scaler := &Scaler{}
	err := scaler.Init(ops.EmptyNodeProto())

	expected := ops.ErrInvalidAttributeCount(2, 0, scaler)
	assert.Equal(t, expected, err)
}

func TestScaler(t *testing.T) {
	tests := []struct {
		scaler          *Scaler
		shape           []int
		backing         []float32
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			&Scaler{
				offset: tensor.New(tensor.WithBacking([]float32{2.0, 25.0, 450.0})),
				scale:  tensor.New(tensor.WithBacking([]float32{0.5, 1.0, 2.0})),
			},
			[]int{2, 3},
			[]float32{1, 20, 300, 4, 50, 600},
			[]int{2, 3},
			[]float32{-0.5, -5.0, -300.0, 1.0, 25.0, 300.0},
		},
		{
			&Scaler{
				offset: tensor.New(tensor.WithBacking([]float32{2.0})),
				scale:  tensor.New(tensor.WithBacking([]float32{0.5})),
			},
			[]int{2, 3},
			[]float32{1, 2, 3, 4, 5, 6},
			[]int{2, 3},
			[]float32{-0.5, 0.0, 0.5, 1.0, 1.5, 2.0},
		},
		{
			&Scaler{
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

func TestInputValidationScaler(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int32{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int64{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, scaler1BaseOpFixture()),
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", scaler1BaseOpFixture()),
		},
	}

	for _, test := range tests {
		scaler := scalerVersions[test.version]()
		validated, err := scaler.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func ScalerOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "offset", Floats: []float32{1.5, 2.5, 3.5}},
			{Name: "scale", Floats: []float32{0.5, 1.0, 2.0}},
		},
	}
}

func scaler1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 1, 1, scalerTypeConstraints, "scaler")
}
