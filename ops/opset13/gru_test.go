package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGruInit(t *testing.T) {
	gru := &GRU{}
	err := gru.Init(GRUOnnxAttributeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, true, gru.linearBeforeReset)
	assert.Equal(t, 5, gru.hiddenSize)
}

func TestGruInitUnkownAttr(t *testing.T) {
	gru := GRU{}
	tests := []struct {
		attr []*onnx.AttributeProto
		err  error
	}{
		{
			[]*onnx.AttributeProto{{Name: "activation_alpha"}},
			ops.ErrInvalidAttribute("activation_alpha", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "activation_beta"}},
			ops.ErrInvalidAttribute("activation_beta", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "direction"}},
			ops.ErrInvalidAttribute("direction", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "clip"}},
			ops.ErrInvalidAttribute("clip", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "activation"}},
			ops.ErrInvalidAttribute("activation", &gru),
		},
		{
			[]*onnx.AttributeProto{{Name: "unknown"}},
			ops.ErrInvalidAttribute("unknown", &gru),
		},
	}

	for _, test := range tests {
		err := gru.Init(test.attr)
		assert.Equal(t, test.err, err)
	}
}

func TestGru(t *testing.T) {
	tests := []struct {
		gru      *GRU
		inputs   ops.InputFixture
		expected []float32
		err      error
	}{
		{
			&GRU{4, true},
			gruInput0,
			[]float32{6.6936556e-03, 8.3446503e-07, 0.0000000e+00, 0.0000000e+00},
			nil,
		},
		{
			&GRU{4, false},
			gruInput0,
			[]float32{6.6936556e-03, 8.3446503e-07, 0.0000000e+00, 0.0000000e+00},
			nil,
		},
		{
			&GRU{4, false},
			gruInput1,
			[]float32{0.44905475, 0.4406946, 0.43368173, 0.42782417},
			nil,
		},
		{
			&GRU{4, false},
			gruInputNoBNoH,
			[]float32{0.24553154, 0.24553154, 0.24553154, 0.24553154},
			nil,
		},
	}

	for _, test := range tests {
		inputs := test.inputs()
		res, err := test.gru.Apply(inputs)
		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expected, res[1].Data())
		}
	}
}

func TestInputValidationGRU(t *testing.T) {
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
			ops.ErrInvalidInputCount(1, &GRU{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", &GRU{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "int", &GRU{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", &GRU{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(2, "int", &GRU{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidInputType(3, "int", &GRU{}),
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
			ops.ErrInvalidInputType(4, "float32", &GRU{}),
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
			ops.ErrInvalidInputType(4, "int", &GRU{}),
		},
	}

	for _, test := range tests {
		gru := &GRU{}
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

func GRUOnnxAttributeProtoFixture() []*onnx.AttributeProto {
	return []*onnx.AttributeProto{
		{Name: "linear_before_reset", I: 1},
		{Name: "hidden_size", I: 5},
	}
}
