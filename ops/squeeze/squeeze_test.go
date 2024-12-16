package squeeze

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
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
		version       int64
		shape         []int
		dimsToDrop    []int64
		expectedShape tensor.Shape
	}{
		{
			13,
			[]int{3, 1, 2},
			[]int64{1},
			[]int{3, 2},
		},
		{
			13,
			[]int{3, 1, 2},
			[]int64{-2},
			[]int{3, 2},
		},
		{
			13,
			[]int{1, 4, 3, 1},
			[]int64{0, -1},
			[]int{4, 3},
		},
		{
			13,
			[]int{1, 4, 3, 1},
			[]int64{0},
			[]int{4, 3, 1},
		},
	}

	for _, test := range tests {
		squeeze := squeezeVersions[test.version]()
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
		version       int64
		shape         []int
		expectedShape tensor.Shape
	}{
		{
			13,
			[]int{3, 1, 2},
			[]int{3, 2},
		},
		{
			13,
			[]int{1, 4, 3, 1},
			[]int{4, 3},
		},
	}

	for _, test := range tests {
		squeeze := squeezeVersions[test.version]()
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
		version  int64
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			nil,
		},
		{
			11,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2), nil},
			nil,
		},
		{
			13,
			[]tensor.Tensor{},
			nil,
			ops.ErrInvalidOptionalInputCount(0, squeeze13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			ops.ErrInvalidOptionalInputCount(3, squeeze13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", squeeze13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		squeeze := squeezeVersions[test.version]()
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

func squeeze13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 2, squeezeTypeConstraints, "squeeze")
}
