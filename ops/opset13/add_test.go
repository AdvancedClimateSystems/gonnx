package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAddInit(t *testing.T) {
	a := &Add{}

	// since 'add' does not have any attributes we pass in nil. This should not
	// fail initializing the add.
	err := a.Init(nil)
	assert.Nil(t, err)
}

func TestAdd(t *testing.T) {
	tests := []struct {
		add      *Add
		backings [][]float32
		shapes   [][]int
		expected []float32
	}{
		{
			&Add{},
			[][]float32{{0, 1, 2, 3}, {1, 1, 1, 1}},
			[][]int{{2, 2}, {2, 2}},
			[]float32{1, 2, 3, 4},
		},
		{
			&Add{},
			[][]float32{{0, 1, 2, 3, 4, 5}, {2, 2, 2, 2, 2, 2}},
			[][]int{{3, 2}, {3, 2}},
			[]float32{2, 3, 4, 5, 6, 7},
		},
		{
			&Add{},
			[][]float32{{0, 1}, {0, 1, 2, 3}},
			[][]int{{2}, {2, 2}},
			[]float32{0, 2, 2, 4},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}

		res, err := test.add.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestAddFail(t *testing.T) {
	inputs := []tensor.Tensor{
		ops.TensorWithBackingFixture([]float32{1, 2, 3, 4}, 2, 2),
		ops.TensorWithBackingFixture([]float32{1, 2, 3}, 3),
	}

	add := &Add{}
	_, err := add.Apply(inputs)
	assert.Equal(
		t,
		err,
		fmt.Errorf(
			ops.MultidirBroadcastErrTemplate,
			[]int{2, 2},
			[]int{3},
			"incompatible dimensions",
		),
	)
}

func TestInputValidationAdd(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("add operator: expected 2 input tensors, got 1"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			fmt.Errorf("add operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		add := &Add{}
		validated, err := add.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
