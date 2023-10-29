package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestPReluInit(t *testing.T) {
	p := &PRelu{}

	// since the prelu does not have any attributes we pass in nil. This should not
	// fail initializing the prelu.
	err := p.Init(nil)
	assert.Nil(t, err)
}

func TestPRelu(t *testing.T) {
	tests := []struct {
		prelu    *PRelu
		backing  []float32
		slope    []float32
		shape    []int
		expected []float32
	}{
		{
			&PRelu{},
			[]float32{-4, -4, -4, -3, -2, -1},
			[]float32{2, 2, 4, 4, 0, 0},
			[]int{3, 2},
			[]float32{-8, -8, -16, -12, 0, 0},
		},
		{
			&PRelu{},
			[]float32{-4, -4, -4, 3, 2, 1},
			[]float32{2, 2, 4, 4, 0, 0},
			[]int{3, 2},
			[]float32{-8, -8, -16, 3, 2, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
			ops.TensorWithBackingFixture(test.slope, test.shape...),
		}
		res, err := test.prelu.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationPRelu(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{},
			fmt.Errorf("prelu operator: expected 2 input tensors, got 0"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			fmt.Errorf("prelu operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		prelu := &PRelu{}
		validated, err := prelu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func BenchmarkPRelu_Apply(b *testing.B) {
	prelu := &PRelu{}
	input := ops.Float32TensorFixture(3, 256, 256)
	slope := ops.Float32TensorFixture(3, 256, 256)
	inputs := []tensor.Tensor{input, slope}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		y, err := prelu.Apply(inputs)
		if err != nil {
			b.Fatal(err)
		}

		_ = y
	}
}
