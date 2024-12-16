package shape

import (
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
		version    int64
		inputShape []int
		expected   []int64
	}{
		{
			1,
			[]int{1, 2, 3, 4},
			[]int64{1, 2, 3, 4},
		},
		{
			13,
			[]int{2, 3},
			[]int64{2, 3},
		},
	}

	for _, test := range tests {
		shape := shapeVersions[test.version]()
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
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]uint32{3, 4}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{3, 4}, 2)},
			nil,
		},
		{
			13,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, shape13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", shape13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		shape := shapeVersions[test.version]()
		validated, err := shape.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func shape13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, shapeTypeConstraints, "shape")
}
