package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLogSoftmaxInit(t *testing.T) {
	l := &LogSoftmax{}

	// since 'logsoftmax' does not have any attributes we pass in nil. This should not
	// fail initializing the logsoftmax.
	err := l.Init(nil)
	assert.Nil(t, err)
}

func TestLogSoftmax(t *testing.T) {
	tests := []struct {
		logsoftmax *LogSoftmax
		backing    []float32
		shape      []int
		expected   []float32
	}{
		{
			&LogSoftmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{-3.4401898, -2.4401898, -1.4401897, -0.44018975},
		},
		{
			&LogSoftmax{
				axis: 1,
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{-3.4401898, -2.4401898, -1.4401897, -0.44018975},
		},
		{
			&LogSoftmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{-1.3132616, -0.31326166, -1.3132616, -0.31326166},
		},
		{
			&LogSoftmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{1, 2, 3},
			[]float32{-2.407606, -1.4076059, -0.40760595, -2.407606, -1.4076059, -0.40760595},
		},
		{
			&LogSoftmax{
				axis: -1,
			},
			[]float32{0, 1, 2, 3},
			[]int{4, 1},
			[]float32{0, 0, 0, 0},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		res, err := test.logsoftmax.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestLogSoftmaxFail(t *testing.T) {
	inputs := []tensor.Tensor{
		ops.TensorWithBackingFixture([]float32{1, 2, 3, 4}, 2, 2),
	}

	logsoftmax := &LogSoftmax{
		axis: 3, // This axis is out of range.
	}
	_, err := logsoftmax.Apply(inputs)
	assert.Equal(
		t,
		err,
		fmt.Errorf(
			ops.AxisOutOfRangeErrTemplate,
			2,
			2,
			3,
		),
	)
}

func TestInputValidationLogSoftmax(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
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
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("logsoftmax operator: expected 1 input tensors, got 2"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			fmt.Errorf("logsoftmax operator: input 0 does not allow type int"),
		},
	}

	for _, test := range tests {
		logsoftmax := &LogSoftmax{axis: -1}
		validated, err := logsoftmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
