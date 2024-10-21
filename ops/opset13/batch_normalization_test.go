package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestBatchNormalizationInit(t *testing.T) {
	b := &BatchNormalization{}

	err := b.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "epsilon", F: 0.001},
			},
		},
	)
	assert.Nil(t, err)

	assert.Equal(t, float32(0.001), b.epsilon)
	assert.True(t, b.testMode)
}

func TestBatchNormalizationInitTrainingMode(t *testing.T) {
	b := &BatchNormalization{}

	err := b.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "epsilon", F: 0.001},
				{Name: "momentum", F: 0.99},
			},
		},
	)
	assert.Equal(t, ops.ErrUnsupportedAttribute("momentum", b), err)

	assert.Equal(t, float32(0.001), b.epsilon)
	assert.Equal(t, float32(0.99), b.momentum)
	assert.False(t, b.testMode)
}

func TestBatchNormalization(t *testing.T) {
	tests := []struct {
		batchNormalization *BatchNormalization
		backings           [][]float32
		shapes             [][]int
		expected           []float32
	}{
		{
			&BatchNormalization{
				epsilon:  1e5,
				momentum: 0.9,
				testMode: true,
			},
			[][]float32{
				{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
				{0.2, 0.3, 0.4},
				{0.1, -0.1, 0.2},
				{4, 8, 12},
				{1, 2, 3},
			},
			[][]int{
				{2, 3, 2, 2},
				{3},
				{3},
				{3},
				{3},
			},
			[]float32{0.097470194, 0.098102644, 0.098735094, 0.09936755, -0.103794694, -0.10284603, -0.10189735, -0.10094868, 0.19494043, 0.19620533, 0.19747022, 0.19873512, 0.10505962, 0.10569207, 0.10632452, 0.10695698, -0.09241061, -0.091461934, -0.09051326, -0.08956459, 0.21011914, 0.21138403, 0.21264893, 0.21391381},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
			ops.TensorWithBackingFixture(test.backings[2], test.shapes[2]...),
			ops.TensorWithBackingFixture(test.backings[3], test.shapes[3]...),
			ops.TensorWithBackingFixture(test.backings[4], test.shapes[4]...),
		}

		res, err := test.batchNormalization.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationBatchNormalization(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, &BatchNormalization{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputType(1, "int", &BatchNormalization{}),
		},
	}

	for _, test := range tests {
		batchNormalization := &BatchNormalization{}
		validated, err := batchNormalization.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
