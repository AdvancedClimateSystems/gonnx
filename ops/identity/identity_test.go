package identity

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestIdentityInit(t *testing.T) {
	i := &Identity{}

	// since 'identity' does not have any attributes we pass in nil. This should not
	// fail initializing the identity.
	err := i.Init(nil)
	assert.Nil(t, err)
}

func TestIdentity(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			13,
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0, 1, 2, 3},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		identity := identityVersions[test.version]()

		res, err := identity.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationIdentity(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			ops.ErrInvalidInputCount(2, identity13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", identity13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		identity := identityVersions[test.version]()
		validated, err := identity.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func identity13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, identityTypeConstraints, "identity")
}
