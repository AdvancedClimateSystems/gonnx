package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestDivInit(t *testing.T) {
	div := &Div{}

	// since the div does not have any attributes we pass in nil. This should not
	// fail initializing the div.
	err := div.Init(nil)
	assert.Nil(t, err)
}

func TestDiv(t *testing.T) {
	tests := []struct {
		div      *Div
		shapes   [][]int
		backings [][]float32
		expected []float32
	}{
		{
			&Div{},
			[][]int{{2, 2}, {2, 2}},
			[][]float32{{10, 10, 10, 10}, {2, 5, 2.5, 1.0}},
			[]float32{5, 2, 4, 10},
		},
		{
			&Div{},
			[][]int{{2, 2}, {2}},
			[][]float32{{1, 1, 1, 1}, {1, 2}},
			[]float32{1, 0.5, 1, 0.5},
		},
		{
			&Div{},
			[][]int{{2, 2}, {1}},
			[][]float32{{1, 1, 1, 1}, {2}},
			[]float32{0.5, 0.5, 0.5, 0.5},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
		}
		res, err := test.div.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationDiv(t *testing.T) {
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
			ops.ErrInvalidInputCount(1, &Div{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputType(0, "int", &Div{}),
		},
	}

	for _, test := range tests {
		div := &Div{}
		validated, err := div.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
