package gru

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGruInit(t *testing.T) {
	gru := GRU{}
	err := gru.Init(GRUOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.0}, gru.activationAlpha)
	assert.Equal(t, []float32{2.0}, gru.activationBeta)
	assert.Equal(t, []string{"sigmoid", "tanh"}, gru.activations)
	assert.Equal(t, gru.direction, ops.Forward)
	assert.Equal(t, 5, gru.hiddenSize)
	assert.Equal(t, true, gru.linearBeforeReset)
}

func TestGruInitUnkownAttr(t *testing.T) {
	gru := GRU{}
	tests := []struct {
		attr []*onnx.AttributeProto
		err  error
	}{
		{
			[]*onnx.AttributeProto{{Name: "clip"}},
			ops.ErrUnsupportedAttribute("clip", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "unknown"}},
			ops.ErrInvalidAttribute("unknown", &gru),
		},
	}

	for _, test := range tests {
		err := gru.Init(&onnx.NodeProto{Attribute: test.attr})
		assert.Equal(t, test.err, err)
	}
}

func TestGru(t *testing.T) {
	tests := []struct {
		version  int64
		node     *onnx.NodeProto
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
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
					{Name: "linear_before_reset", I: 1},
				},
			},
			gruInput0,
			[]float32{6.6936556e-03, 8.3446503e-07, 0.0000000e+00, 0.0000000e+00},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
					{Name: "linear_before_reset", I: 0},
				},
			},
			gruInput0,
			[]float32{6.6936556e-03, 8.3446503e-07, 0.0000000e+00, 0.0000000e+00},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
					{Name: "linear_before_reset", I: 0},
				},
			},
			gruInput1,
			[]float32{0.44905475, 0.4406946, 0.43368173, 0.42782417},
			nil,
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "activation_alpha", Floats: []float32{}},
					{Name: "activation_beta", Floats: []float32{}},
					{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh")}},
					{Name: "direction", S: []byte("forward")},
					{Name: "hidden_size", I: 4},
					{Name: "linear_before_reset", I: 0},
				},
			},
			gruInputNoBNoH,
			[]float32{0.24553154, 0.24553154, 0.24553154, 0.24553154},
			nil,
		},
	}

	for _, test := range tests {
		inputs := test.inputs()

		gru := gruVersions[test.version]()
		err := gru.Init(test.node)
		assert.Nil(t, err)

		res, err := gru.Apply(inputs)
		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expected, res[1].Data())
		}
	}
}

func TestInputValidationGRU(t *testing.T) {
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
			},
			nil,
		},
		{
			7,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, gru7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", gru7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "int", gru7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", gru7BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(2, "int", gru7BaseOpFixture()),
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
			ops.ErrInvalidInputType(3, "int", gru7BaseOpFixture()),
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
			ops.ErrInvalidInputType(4, "float32", gru7BaseOpFixture()),
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
			ops.ErrInvalidInputType(5, "int", gru7BaseOpFixture()),
		},
	}

	for _, test := range tests {
		gru := gruVersions[test.version]()
		validated, err := gru.ValidateInputs(test.inputs)

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

func gruInput0() []tensor.Tensor {
	shapes := [][]int{{2, 1, 3}, {1, 12, 3}, {1, 12, 4}, {1, 24}, {1, 1, 4}}
	inputs := []tensor.Tensor{
		ops.Float32TensorFixture(shapes[0]...),
		ops.Float32TensorFixture(shapes[1]...),
		ops.Float32TensorFixture(shapes[2]...),
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(shapes[3]...)), shapes[3]...),
		nil,
		ops.TensorWithBackingFixture(ops.Zeros(ops.NElements(shapes[4]...)), shapes[4]...),
	}

	return inputs
}

func gruInput1() []tensor.Tensor {
	shps := [][]int{{10, 1, 3}, {1, 12, 3}, {1, 12, 4}, {1, 24}, {1, 1, 4}}
	inputs := []tensor.Tensor{
		ops.Float32TensorFixture(shps[0]...),
		ops.TensorWithBackingFixture(ops.Full(ops.NElements(shps[1]...), 0.2), shps[1]...),
		ops.TensorWithBackingFixture(ops.Full(ops.NElements(shps[2]...), 0.5), shps[2]...),
		ops.TensorWithBackingFixture(ops.Arange(ops.NElements(shps[3]...), 0.10), shps[3]...),
		nil,
		ops.TensorWithBackingFixture(ops.Full(ops.NElements(shps[4]...), 0.4), shps[4]...),
	}

	return inputs
}

func gruInputNoBNoH() []tensor.Tensor {
	shps := [][]int{{10, 1, 3}, {1, 12, 3}, {1, 12, 4}, {1, 24}, {1, 1, 4}}
	inputs := []tensor.Tensor{
		ops.Float32TensorFixture(shps[0]...),
		ops.TensorWithBackingFixture(ops.Full(ops.NElements(shps[1]...), 0.2), shps[1]...),
		ops.TensorWithBackingFixture(ops.Full(ops.NElements(shps[2]...), 0.5), shps[2]...),
		nil,
		nil,
		nil,
	}

	return inputs
}

func GRUOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "activation_alpha", Floats: []float32{1.0}},
			{Name: "activation_beta", Floats: []float32{2.0}},
			{Name: "activations", Strings: [][]byte{[]byte("sigmoid"), []byte("tanh")}},
			{Name: "direction", S: []byte("forward")},
			{Name: "hidden_size", I: 5},
			{Name: "linear_before_reset", I: 1},
		},
	}
}

func gru7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(7, 3, 6, gruTypeConstraints, "gru")
}
