package lstm

import (
	"math/rand"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLSTMInit(t *testing.T) {
	lstm := &LSTM{}
	err := lstm.Init(LSTMOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.0}, lstm.activationAlpha)
	assert.Equal(t, []float32{2.0}, lstm.activationBeta)
	assert.Equal(t, []string{"sigmoid", "tanh", "relu"}, lstm.activations)
	assert.Equal(t, ops.Forward, lstm.direction)
	assert.Equal(t, 5, lstm.hiddenSize)
	assert.Equal(t, false, lstm.inputForget)
	assert.Equal(t, []string{"Y", "Y_h"}, lstm.outputs)
}

func TestLSTMInitUnkownAttr(t *testing.T) {
	lstm := LSTM{}
	tests := []struct {
		attr []*onnx.AttributeProto
		err  error
	}{
		{
			[]*onnx.AttributeProto{{Name: "clip"}},
			ops.ErrUnsupportedAttribute("clip", &lstm),
		},
		{
			[]*onnx.AttributeProto{{Name: "unknown"}},
			ops.ErrInvalidAttribute("unknown", &lstm),
		},
	}

	for _, test := range tests {
		err := lstm.Init(&onnx.NodeProto{Attribute: test.attr})
		assert.Equal(t, test.err, err)
	}
}

func TestLSTM(t *testing.T) {
	tests := []struct {
		version  int64
		attrs    *onnx.NodeProto
		inputs   ops.InputFixture
		expected []float32
		err      error
	}{
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
				},
				Output: []string{"Y", "Y_h", "Y_c"},
			},
			lstmInput0,
			[]float32{0.9159305, 0.9356764, 0.87070554, 0.84180677},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("relu")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
				},
				Output: []string{"Y", "Y_h", "Y_c"},
			},
			lstmInput0,
			[]float32{1.7530097, 1.7829735, 1.6231446, 1.5197954},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("relu")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
				},
				Output: []string{"Y", "Y_h", "Y_c"},
			},
			lstmInput1,
			[]float32{10.598255, 10.547241, 10.214846, 10.267471},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("relu")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
				},
				Output: []string{"Y", "Y_h", "Y_c"},
			},
			lstmInputNoBNoH,
			[]float32{8.276371, 8.291079, 8.161418, 7.7900877},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
				},
				Output: []string{"Y", "Y_h", "Y_c"},
			},
			lstmInputPeepholes,
			[]float32{0.99891853, 0.99994266, 0.9995524, 0.99171203},
			nil,
		},
	}

	for _, test := range tests {
		inputs := test.inputs()

		lstm := lstmVersions[test.version]()
		err := lstm.Init(test.attrs)
		assert.Nil(t, err)

		res, err := lstm.Apply(inputs)
		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expected, res[1].Data())
		}
	}
}

func TestInputValidationLSTM(t *testing.T) {
	tests := []struct {
		version  int64
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			nil,
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				nil,
				nil,
				nil,
				nil,
				nil,
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(2, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(3, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(4, "float32", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(5, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(6, "int", lstm7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(7, "int", lstm7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		lstm := lstmVersions[test.version]()
		validated, err := lstm.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			if test.expected != nil {
				assert.Equal(t, test.expected, validated)
			} else {
				assert.Equal(t, test.inputs, validated)
			}
		}
	}
}

func lstmInput0() []tensor.Tensor {
	r := rand.New(rand.NewSource(10))

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(r, 2, 1, 3),
		// Input W: (num_directions, 4 * hidden_size, input_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 3),
		// Input R: (num_directions, 4 * hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 4),
		// Input B: (num_directions, 8 * hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 32),
		// Input sequence_lens: not supported.
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size).
		ops.TensorWithBackingFixture(ops.Zeros(4), 1, 1, 4),
		// Input initial_c: (num_directions, batch_size, hidden_size).
		ops.TensorWithBackingFixture(ops.Zeros(4), 1, 1, 4),
		// Input P: peephole weights.
		nil,
	}
}

func lstmInput1() []tensor.Tensor {
	r := rand.New(rand.NewSource(11))

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(r, 10, 1, 3),
		// Input W: (num_directions, 4 * hidden_size, input_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 3),
		// Input R: (num_directions, 4 * hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 4),
		// Input B: (num_directions, 8 * hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 32),
		// Input sequence_lens: not supported.
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 1, 4),
		// Input initial_c: (num_directions, batch_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 1, 4),
		// Input P: peephole weights.
		nil,
	}
}

func lstmInputNoBNoH() []tensor.Tensor {
	r := rand.New(rand.NewSource(12))

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(r, 10, 1, 3),
		// Input W: (num_directions, 4 * hidden_size, input_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 3),
		// Input R: (num_directions, 4 * hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 4),
		// Input B.
		nil,
		// Input sequence_lens: not supported.
		nil,
		// Input initial_h.
		nil,
		// Input initial_c.
		nil,
		// Input P: peephole weights.
		nil,
	}
}

func lstmInputPeepholes() []tensor.Tensor {
	r := rand.New(rand.NewSource(13))

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(r, 10, 1, 3),
		// Input W: (num_directions, 4 * hidden_size, input_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 3),
		// Input R: (num_directions, 4 * hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 16, 4),
		// Input B.
		nil,
		// Input sequence_lens: not supported.
		nil,
		// Input initial_h.
		nil,
		// Input initial_c.
		nil,
		// Input P: (num_directions, 3 * hidden_size).
		ops.RandomFloat32TensorFixture(r, 1, 12),
	}
}

func LSTMOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "activation_alpha", Floats: []float32{1.0}},
			{Name: "activation_beta", Floats: []float32{2.0}},
			{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh"), []byte("relu")}},
			{Name: "direction", S: []byte("forward")},
			{Name: "hidden_size", I: 5},
			{Name: "input_forget", I: 0},
		},
		Output: []string{"Y", "Y_h"},
	}
}

func lstm7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 3, 8, lstmTypeConstraints, "lstm")
}
