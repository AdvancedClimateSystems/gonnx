package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestFlattenInit(t *testing.T) {
	f := &Flatten{}

	err := f.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 2}}})
	assert.Nil(t, err)

	assert.Equal(t, 2, f.axis)
}

func TestFlatten(t *testing.T) {
	tests := []struct {
		flatten       *Flatten
		backing       []float32
		shape         []int
		expectedShape tensor.Shape
	}{
		{
			&Flatten{},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]int{1, 4},
		},
		{
			&Flatten{},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]int{1, 6},
		},
		{
			&Flatten{axis: 1},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7},
			[]int{2, 2, 2},
			[]int{2, 4},
		},
		{
			&Flatten{axis: 2},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7},
			[]int{2, 2, 2},
			[]int{4, 2},
		},
		{
			&Flatten{axis: -1},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7},
			[]int{2, 2, 2},
			[]int{4, 2},
		},
		{
			&Flatten{axis: -2},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7},
			[]int{2, 2, 2},
			[]int{2, 4},
		},
		{
			&Flatten{axis: -3},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			[]int{3, 2, 3},
			[]int{1, 18},
		},
		{
			&Flatten{axis: 2},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			[]int{3, 2, 3},
			[]int{6, 3},
		},
		{
			&Flatten{axis: 1},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			[]int{3, 2, 3},
			[]int{3, 6},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.flatten.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
	}
}

func TestInputValidationFlatten(t *testing.T) {
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
			ops.ErrInvalidInputCount(2, &Flatten{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Flatten{}),
		},
	}

	for _, test := range tests {
		flatten := &Flatten{}
		validated, err := flatten.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
