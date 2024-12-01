package cast

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCast13Init(t *testing.T) {
	c := &Cast13{}

	err := c.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "to", I: 1}}})
	assert.Nil(t, err)
	assert.Equal(t, int32(1), c.to)
}

func TestCast13(t *testing.T) {
	tests := []struct {
		cast     *Cast13
		backing  interface{}
		shape    []int
		to       int64
		expected interface{}
	}{
		{
			&Cast13{},
			[]float32{1.0, 1.0},
			[]int{2},
			11,
			[]float64{1.0, 1.0},
		},
		{
			&Cast13{},
			[]float32{1.3, 1.8},
			[]int{2},
			4,
			[]uint16{1, 1},
		},
		{
			&Cast13{},
			[]int8{1, 1},
			[]int{2},
			1,
			[]float32{1.0, 1.0},
		},
		{
			&Cast13{},
			[]int64{1, 1},
			[]int{2},
			11,
			[]float64{1.0, 1.0},
		},
		{
			&Cast13{},
			[]float64{1.4, 1.5},
			[]int{2},
			3,
			[]int8{1, 1},
		},
	}

	for _, test := range tests {
		_ = test.cast.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "to", I: test.to}}})
		inputs := []tensor.Tensor{ops.TensorWithBackingFixture(test.backing, test.shape...)}

		res, err := test.cast.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationCast13(t *testing.T) {
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
			ops.ErrInvalidInputCount(2, &Cast13{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{true, false}, 2),
			},
			ops.ErrInvalidInputType(0, "bool", &Cast13{}),
		},
	}

	for _, test := range tests {
		cast := &Cast13{}
		validated, err := cast.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
