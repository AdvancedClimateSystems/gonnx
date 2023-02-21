package opset13

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gitlab.advancedclimate.nl/smartbase/software/core/airgo/gonnx/ops"
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
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint8{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint16{1, 2}, 2),
			},
			nil,
		},
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
				ops.TensorWithBackingFixture([]int8{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int16{1, 2}, 2),
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
			[]tensor.Tensor{},
			fmt.Errorf("abs operator: expected 1 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("abs operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		abs := &Abs{}
		validated, err := abs.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
