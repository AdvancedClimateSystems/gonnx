package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestShapeInit(t *testing.T) {
	s := &Shape{}

	// since 'shape' does not have any attributes we pass in nil. This should not
	// fail initializing the shape operator.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestShape(t *testing.T) {
	tests := []struct {
		inputShape []int
		expected   []int64
	}{
		{
			[]int{1, 2, 3, 4},
			[]int64{1, 2, 3, 4},
		},
		{
			[]int{2, 3},
			[]int64{2, 3},
		},
	}

	for _, test := range tests {
		shape := &Shape{}
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.inputShape...),
		}

		res, err := shape.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationShape(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{3, 4}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{3, 4}, 2)},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("shape operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			fmt.Errorf("shape operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		shape := &Shape{}
		validated, err := shape.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
