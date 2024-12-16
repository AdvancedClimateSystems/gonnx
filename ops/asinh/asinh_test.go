package asinh

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAsinhInit(t *testing.T) {
	c := &Asinh{}

	// since 'asinh' does not have any attributes we pass in nil. This should not
	// fail initializing the asinh.
	err := c.Init(nil)
	assert.Nil(t, err)
}

func TestAsinh(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			9,
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{0.8813736, 1.4436355, 1.8184465, 2.0947125},
		},
		{
			9,
			[]float32{1, 2, 3, 4},
			[]int{1, 4},
			[]float32{0.8813736, 1.4436355, 1.8184465, 2.0947125},
		},
		{
			9,
			[]float32{2, 2, 2, 2},
			[]int{1, 4},
			[]float32{1.4436355, 1.4436355, 1.4436355, 1.4436355},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		asinh := asinhVersions[test.version]()

		res, err := asinh.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAsinh(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			9,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, asinh9BaseOpFixture()),
		},
		{
			9,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", asinh9BaseOpFixture()),
		},
	}

	for _, test := range tests {
		asinh := asinhVersions[test.version]()
		validated, err := asinh.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func asinh9BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(9, 1, 1, asinhTypeConstraints, "asinh")
}
