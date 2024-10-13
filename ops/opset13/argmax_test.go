package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestArgMaxInit(t *testing.T) {
	a := &ArgMax{}

	err := a.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "axis", I: 2},
				{Name: "keepdims", I: 0},
				{Name: "select_last_index", I: 0},
			},
		},
	)
	assert.Nil(t, err)

	assert.Equal(t, 2, a.axis)
	assert.Equal(t, false, a.keepDims)
	assert.Equal(t, false, a.selectLastIndex)
}

func TestArgMax(t *testing.T) {
	tests := []struct {
		argmax        *ArgMax
		backing       []float32
		shape         []int
		expectedShape tensor.Shape
		expectedData  []int64
	}{
		{
			&ArgMax{axis: 0, keepDims: true},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int{1, 2},
			[]int64{1, 1},
		},
		{
			&ArgMax{axis: -1, keepDims: true},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int{2, 1},
			[]int64{1, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.argmax.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedData, res[0].Data())
	}
}

func TestInputValidationArgMax(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, &ArgMax{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &ArgMax{}),
		},
	}

	for _, test := range tests {
		argmax := &ArgMax{}
		validated, err := argmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
