package abs

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAbsInit(t *testing.T) {
	a := &Abs{}

	// since 'abs' does not have any attributes we pass in nil. This should not
	// fail initializing the abs.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAbs(t *testing.T) {
	tests := []struct {
		abs      *Abs
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			&Abs{},
			[]float32{-2, -1, 0, 1},
			[]int{2, 2},
			[]float32{2, 1, 0, 1},
		},
		{
			&Abs{},
			[]float32{1, 3, 4, 5},
			[]int{1, 4},
			[]float32{1, 3, 4, 5},
		},
		{
			&Abs{},
			[]float32{-1, -1, -1, -1},
			[]int{1, 4},
			[]float32{1, 1, 1, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.abs.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationAbs(t *testing.T) {
	tests := []struct {
		inputs  []tensor.Tensor
		err     error
		version int64
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint8{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint16{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int8{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int16{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
			6,
		},
		{
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, ops.NewBaseOperator(6, 1, 1, absTypeConstraint, "abs")),
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", ops.NewBaseOperator(6, 1, 1, absTypeConstraint, "abs")),
			6,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint8{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint16{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int8{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int16{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
			13,
		},
		{
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, ops.NewBaseOperator(13, 1, 1, absTypeConstraint, "abs")),
			13,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", ops.NewBaseOperator(13, 1, 1, absTypeConstraint, "abs")),
			13,
		},
	}

	for _, test := range tests {
		abs := absVersions[test.version]()
		validated, err := abs.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		assert.Equal(t, test.inputs, validated)
	}
}
