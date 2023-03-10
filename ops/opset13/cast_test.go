package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCastInit(t *testing.T) {
	c := &Cast{}

	err := c.Init([]*onnx.AttributeProto{{Name: "to", I: 1}})
	assert.Nil(t, err)
	assert.Equal(t, int32(1), c.to)
}

func TestCast(t *testing.T) {
	tests := []struct {
		cast     *Cast
		backing  interface{}
		shape    []int
		to       int64
		expected interface{}
	}{
		{
			&Cast{},
			[]float32{1.0, 1.0},
			[]int{2},
			11,
			[]float64{1.0, 1.0},
		},
		{
			&Cast{},
			[]float32{1.3, 1.8},
			[]int{2},
			4,
			[]uint16{1, 1},
		},
		{
			&Cast{},
			[]int8{1, 1},
			[]int{2},
			1,
			[]float32{1.0, 1.0},
		},
		{
			&Cast{},
			[]int64{1, 1},
			[]int{2},
			11,
			[]float64{1.0, 1.0},
		},
		{
			&Cast{},
			[]float64{1.4, 1.5},
			[]int{2},
			3,
			[]int8{1, 1},
		},
	}

	for _, test := range tests {
		test.cast.Init([]*onnx.AttributeProto{{Name: "to", I: test.to}})
		inputs := []tensor.Tensor{ops.TensorWithBackingFixture(test.backing, test.shape...)}

		res, err := test.cast.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationCast(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			fmt.Errorf("cast operator: expected 1 input tensors, got 2"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{true, false}, 2),
			},
			fmt.Errorf("cast operator: input 0 does not allow type bool"),
		},
	}

	for _, test := range tests {
		cast := &Cast{}
		validated, err := cast.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
