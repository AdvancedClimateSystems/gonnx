package opset13

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"
	"gorgonia.org/tensor"
)

func TestSqueezeInit(t *testing.T) {
	s := &Squeeze{}

	// since the squeeze does not have any attributes we pass in nil. This should not
	// fail initializing the squeeze.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSqueezeCustomDims(t *testing.T) {
	tests := []struct {
		shape         []int
		dimsToDrop    []int64
		expectedShape tensor.Shape
	}{
		{
			[]int{3, 1, 2},
			[]int64{1},
			[]int{3, 2},
		},
		{
			[]int{3, 1, 2},
			[]int64{-2},
			[]int{3, 2},
		},
		{
			[]int{1, 4, 3, 1},
			[]int64{0, -1},
			[]int{4, 3},
		},
		{
			[]int{1, 4, 3, 1},
			[]int64{0},
			[]int{4, 3, 1},
		},
	}

	for _, test := range tests {
		squeeze := &Squeeze{}
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.shape...),
			ops.TensorWithBackingFixture(test.dimsToDrop, len(test.dimsToDrop)),
		}

		res, err := squeeze.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
	}
}

func TestSqueezeNoDims(t *testing.T) {
	tests := []struct {
		shape         []int
		expectedShape tensor.Shape
	}{
		{
			[]int{3, 1, 2},
			[]int{3, 2},
		},
		{
			[]int{1, 4, 3, 1},
			[]int{4, 3},
		},
	}

	for _, test := range tests {
		squeeze := &Squeeze{}
		inputs := []tensor.Tensor{ops.Float32TensorFixture(test.shape...), nil}

		res, err := squeeze.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
	}
}

func TestGetDimsToSqueezeFromNode(t *testing.T) {
	tests := []struct {
		nDims       int
		squeezeDims []int64
		expected    []int
	}{
		{
			2,
			[]int64{3, 2},
			[]int{3, 2},
		},
		{
			2,
			[]int64{-1},
			[]int{1},
		},
	}

	for _, test := range tests {
		input := ops.TensorWithBackingFixture(test.squeezeDims, len(test.squeezeDims))
		dimsToSqueeze, err := getDimsToSqueezeFromTensor(input, test.nDims)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, dimsToSqueeze)
	}
}

func TestGetDimsToSqueezeFromShape(t *testing.T) {
	res := getDimsToSqueezeFromShape([]int{1, 4, 3, 1, 5})
	assert.Equal(t, []int{0, 3}, res)
}

func TestGetNewShape(t *testing.T) {
	res := getNewShape([]int{1, 4, 1, 3}, []int{0, 2})
	assert.Equal(t, []int{4, 3}, res)
}

func TestKeepDim(t *testing.T) {
	assert.Equal(t, false, keepDim(2, []int{1, 2}))
	assert.Equal(t, true, keepDim(2, []int{1, 3}))
	assert.Equal(t, false, keepDim(0, []int{0}))
	assert.Equal(t, true, keepDim(0, []int{1, 3}))
}

func TestInputValidationSqueeze(t *testing.T) {
	tests := []struct {
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2), nil},
			nil,
		},
		{
			[]tensor.Tensor{},
			nil,
			fmt.Errorf("squeeze operator: expected 1-2 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			fmt.Errorf("squeeze operator: expected 1-2 input tensors, got 3"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			fmt.Errorf("squeeze operator: input 1 does not allow type int"),
		},
	}

	for _, test := range tests {
		squeeze := &Squeeze{}
		validated, err := squeeze.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			if test.expected != nil {
				assert.Equal(t, test.expected, validated)
			} else {
				assert.Equal(t, test.inputs, validated)
			}
		}
	}
}
