package logsoftmax

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
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
		version  int64
		attrs    *onnx.NodeProto
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: -1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{-3.4401898, -2.4401898, -1.4401897, -0.44018975},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{1, 4},
			[]float32{-3.4401898, -2.4401898, -1.4401897, -0.44018975},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: -1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{-1.3132616, -0.31326166, -1.3132616, -0.31326166},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: -1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{1, 2, 3},
			[]float32{-2.407606, -1.4076059, -0.40760595, -2.407606, -1.4076059, -0.40760595},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: -1},
				},
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

		logsoftmax := logSoftmaxVersions[test.version]()
		err := logsoftmax.Init(test.attrs)
		assert.Nil(t, err)

		res, err := logsoftmax.Apply(inputs)
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
		ops.ErrAxisOutOfRange(-2, 2, 3),
	)
}

func TestInputValidationLogSoftmax(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
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
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(2, logSoftmax13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", logSoftmax13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		logsoftmax := logSoftmaxVersions[test.version]()

		validated, err := logsoftmax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func logSoftmax13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, logSoftmaxTypeConstraints, "logsoftmax")
}
