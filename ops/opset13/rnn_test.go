package opset13

import (
	"math/rand"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestRNNInit(t *testing.T) {
	rnn := &RNN{}
	err := rnn.Init(RNNOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.0}, rnn.activationAlpha)
	assert.Equal(t, []float32{2.0}, rnn.activationBeta)
	assert.Equal(t, []string{"sigmoid"}, rnn.activations)
	assert.Equal(t, RNNDirection("forward"), rnn.direction)
	assert.Equal(t, 5, rnn.hiddenSize)
}

func TestRNNInitUnsupportedAttr(t *testing.T) {
	rnn := RNN{}
	err := rnn.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "clip"}}})
	assert.Equal(t, err, ops.ErrUnsupportedAttribute("clip", &rnn))
}

func TestRNNInitUnknownAttr(t *testing.T) {
	rnn := RNN{}
	err := rnn.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "unknown"}}})
	assert.Equal(t, err, ops.ErrInvalidAttribute("unknown", &rnn))
}

func TestRNN(t *testing.T) {
	tests := []struct {
		rnn      *RNN
		inputs   ops.InputFixture
		expected []float32
		err      error
	}{
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"tanh"},
				direction:       Forward,
				hiddenSize:      4,
			},
			rnnInput0,
			[]float32{0.78036773, 0.97858655, 0.94110376, 0.90722954},
			nil,
		},
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"sigmoid"},
				direction:       Forward,
				hiddenSize:      4,
			},
			rnnInput0,
			[]float32{0.82048327, 0.922734, 0.89050114, 0.8620579},
			nil,
		},
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"relu"},
				direction:       Forward,
				hiddenSize:      4,
			},
			rnnInput0,
			[]float32{1.0667435, 2.328037, 1.7986122, 1.545068},
			nil,
		},
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"tanh"},
				direction:       Forward,
				hiddenSize:      10,
			},
			rnnInput1,
			[]float32{0.99996024, 0.9999855, 0.99998087, 0.9999288, 0.9997511, 0.99918234, 0.99999964, 0.9999981, 0.9997658, 0.9999618, 0.9998762, 0.9999353, 0.9999194, 0.9999428, 0.9997284, 0.9982606, 0.999999, 0.9999897, 0.99964744, 0.9998234, 0.99997497, 0.9999893, 0.9999906, 0.9999812, 0.99983937, 0.99967873, 0.9999998, 0.9999965, 0.9999516, 0.9999541},
			nil,
		},
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"tanh"},
				direction:       Forward,
				hiddenSize:      4,
			},
			rnnInputNoB,
			// Same values as first test, but B is initialized automatically.
			[]float32{0.78036773, 0.97858655, 0.94110376, 0.90722954},
			nil,
		},
		{
			&RNN{
				activationAlpha: []float32{},
				activationBeta:  []float32{},
				activations:     []string{"tanh"},
				direction:       Forward,
				hiddenSize:      4,
			},
			rnnInputNoBNoH,
			// Same values as first test, but B and H are initialized automatically.
			[]float32{0.78036773, 0.97858655, 0.94110376, 0.90722954},
			nil,
		},
	}

	for _, test := range tests {
		inputs := test.inputs()
		res, err := test.rnn.Apply(inputs)
		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expected, res[1].Data())
		}
	}
}

func TestInputValidationRNN(t *testing.T) {
	tests := []struct {
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			nil,
		},
		{
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
			},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "int", &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(2, "int", &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(3, "int", &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(4, "float32", &RNN{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(5, "int", &RNN{}),
		},
	}

	for _, test := range tests {
		rnn := &RNN{}
		validated, err := rnn.ValidateInputs(test.inputs)

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

func rnnInput0() []tensor.Tensor {
	rand.Seed(13)

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(2, 1, 3),
		// Input W: (num_directions, hidden_size, input_size).
		ops.RandomFloat32TensorFixture(1, 4, 3),
		// Input R: (num_directions, hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(1, 4, 4),
		// Input B: (num_directions, 2 * hidden_size)
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(1, 8)), 1, 8),
		// Input sequence_lens: not supported
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size)
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(1, 1, 4)), 1, 1, 4),
	}
}

func rnnInput1() []tensor.Tensor {
	rand.Seed(13)

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(10, 3, 4),
		// Input W: (num_directions, hidden_size, input_size).
		ops.RandomFloat32TensorFixture(1, 10, 4),
		// Input R: (num_directions, hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(1, 10, 10),
		// Input B: (num_directions, 2 * hidden_size)
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(1, 20)), 1, 20),
		// Input sequence_lens: not supported
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size)
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(1, 3, 10)), 1, 3, 10),
	}
}

func rnnInputNoB() []tensor.Tensor {
	rand.Seed(13)

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(2, 1, 3),
		// Input W: (num_directions, hidden_size, input_size).
		ops.RandomFloat32TensorFixture(1, 4, 3),
		// Input R: (num_directions, hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(1, 4, 4),
		// Input B: not provided.
		nil,
		// Input sequence_lens: not supported
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size)
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(1, 1, 4)), 1, 1, 4),
	}
}

func rnnInputNoBNoH() []tensor.Tensor {
	rand.Seed(13)

	return []tensor.Tensor{
		// Input X: (sequence_length, batch_size, input_size).
		ops.RandomFloat32TensorFixture(2, 1, 3),
		// Input W: (num_directions, hidden_size, input_size).
		ops.RandomFloat32TensorFixture(1, 4, 3),
		// Input R: (num_directions, hidden_size, hidden_size).
		ops.RandomFloat32TensorFixture(1, 4, 4),
		// Input B: not provided.
		nil,
		// Input sequence_lens: not supported
		nil,
		// Input initial_h: (num_directions, batch_size, hidden_size)
		nil,
	}
}

func RNNOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "activation_alpha", Floats: []float32{1.0}},
			{Name: "activation_beta", Floats: []float32{2.0}},
			{Name: "activations", Strings: [][]byte{[]byte("sigmoid")}},
			{Name: "direction", S: []byte("forward")},
			{Name: "hidden_size", I: 5},
		},
	}
}
