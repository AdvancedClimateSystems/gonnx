package gonnx

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestModel(t *testing.T) {
	tests := []struct {
		path     string
		input    Tensors
		expected Tensors
		err      error
	}{
		{
			"./sample_models/onnx_models/mlp.onnx",
			tensorsFixture(
				[]string{"data_input"},
				[][]int{{2, 3}},
				[][]float32{rangeFloat(6)},
			),
			tensorsFixture(
				[]string{"preds"},
				[][]int{{2, 2}},
				[][]float32{{-0.056310713, -1.1901507, -1.5961288, -3.3445296}},
			),
			nil,
		},
		{
			"./sample_models/onnx_models/mlp.onnx",
			tensorsFixture(
				[]string{"data_input"},
				[][]int{{2, 4, 2}},
				[][]float32{rangeFloat(16)},
			),
			nil,
			ErrInvalidShape([]onnx.Dim{{IsDynamic: true, Name: "batch_size", Size: 0}, {IsDynamic: false, Name: "", Size: 3}}, []int{2, 4, 2}),
		},
		{
			"./sample_models/onnx_models/mlp.onnx",
			tensorsFixture(
				[]string{"unknown_input"},
				[][]int{{2, 3}},
				[][]float32{rangeFloat(6)},
			),
			nil,
			ErrModel("tensor: %v not found", "data_input"),
		},
		{
			"./sample_models/onnx_models/gru.onnx",
			tensorsFixture(
				[]string{"data_input", "init_hidden"},
				[][]int{{1, 30, 3}, {1, 1, 5}},
				[][]float32{rangeFloat(90), rangeZeros(5)},
			),
			tensorsFixture(
				[]string{"preds", "hidden_out"},
				[][]int{{1, 30, 5}, {1, 1, 5}},
				[][]float32{expectedGruPredsOut(), expectedGruHiddenOut()},
			),
			nil,
		},
		{
			"./sample_models/onnx_models/scaler.onnx",
			tensorsFixture(
				[]string{"X"},
				[][]int{{2, 3}},
				[][]float32{{1.0, 10.0, 100.0, 1.5, 13.0, 120.0}},
			),
			tensorsFixture(
				[]string{"variable"},
				[][]int{{2, 3}},
				[][]float32{
					{-0.10153462, -0.15617376, -0.73914559, 1.67532117, 1.48365074, 1.91419756},
				},
			),
			nil,
		},
		{
			"./sample_models/onnx_models/ndm.onnx",
			tensorsFixture(
				[]string{"sensor_input", "setpoint_input"},
				[][]int{{1, 18, 4}, {1, 1}},
				[][]float32{rangeFloat(72), rangeFloat(1)},
			),
			tensorsFixture(
				[]string{"optimal_supply_temp"},
				[][]int{{1, 1}},
				[][]float32{{0.89400756}},
			),
			nil,
		},
	}

	for _, test := range tests {
		fmt.Println(test.path)
		model, err := NewModelFromFile(test.path)
		assert.Nil(t, err)

		outputs, err := model.Run(test.input)

		assert.Equal(t, test.err, err)

		if test.expected == nil {
			assert.Nil(t, outputs)
		} else {
			for outputName := range test.expected {
				expectedTensor := test.expected[outputName]
				actualTensor := outputs[outputName]
				assert.InDeltaSlice(t, expectedTensor.Data(), actualTensor.Data(), 0.00001)
			}
		}
	}
}

func TestModelIOUtil(t *testing.T) {
	model, err := NewModelFromFile("./sample_models/onnx_models/mlp.onnx")
	assert.Nil(t, err)

	expectedInputShapes := onnx.Shapes{
		"data_input": []onnx.Dim{
			{IsDynamic: true, Name: "batch_size", Size: 0},
			{IsDynamic: false, Name: "", Size: 3},
		},
	}

	assert.Equal(t, []string{"data_input"}, model.InputNames())
	assert.Equal(t, expectedInputShapes, model.InputShapes())

	expectedOutputShapes := onnx.Shapes{
		"preds": []onnx.Dim{
			{IsDynamic: true, Name: "batch_size", Size: 0},
			{IsDynamic: false, Name: "", Size: 2},
		},
	}

	assert.Equal(t, []string{"preds"}, model.OutputNames())
	assert.Equal(t, expectedOutputShapes, model.OutputShapes())
	assert.Equal(t, expectedOutputShapes["preds"], model.OutputShape("preds"))

	assert.Equal(
		t,
		[]string{"layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"},
		model.ParamNames(),
	)

	assert.True(t, model.hasInput("data_input"))
	assert.False(t, model.hasInput("fail"))
}

func TestInputDimSize(t *testing.T) {
	model, err := NewModelFromFile("./sample_models/onnx_models/mlp.onnx")
	assert.Nil(t, err)

	dimSize, err := model.InputDimSize("data_input", 1)
	assert.Nil(t, err)
	assert.Equal(t, 3, dimSize)
}

func TestInputDimSizeInvalidInput(t *testing.T) {
	model, err := NewModelFromFile("./sample_models/onnx_models/mlp.onnx")
	assert.Nil(t, err)

	_, err = model.InputDimSize("swagger", 0)

	assert.Equal(t, ErrModel("input %v does not exist", "swagger"), err)
}

// tensorsFixture creates Tensors with the given names shapes and backings. This is useful for
// providing a model with inputs and checking it's outputs.
func tensorsFixture(names []string, shapes [][]int, backing [][]float32) Tensors {
	res := make(Tensors, len(names))
	for i, name := range names {
		res[name] = tensor.New(
			tensor.WithShape(shapes[i]...),
			tensor.WithBacking(backing[i]),
		)
	}

	return res
}

func rangeFloat(size int) []float32 {
	res := make([]float32, size)
	for i := 0; i < size; i++ {
		res[i] = float32(i)
	}

	return res
}

func rangeZeros(size int) []float32 {
	res := make([]float32, size)
	for i := range res {
		res[i] = 0.0
	}

	return res
}

func expectedGruHiddenOut() []float32 {
	return []float32{0.45711097, 1, 0.9258882, -1, 1}
}

func expectedGruPredsOut() []float32 {
	return []float32{
		0.254439, 0.39027894, 0.12178477, 0.24339758, 0.39764592,
		0.3930065, 0.7781081, 0.41358948, 0.018615374, 0.9664475,
		0.43873432, 0.94687027, 0.5921345, -0.32937312, 0.99975467,
		0.45136297, 0.98945767, 0.6988256, -0.5221157, 0.99999934,
		0.45525378, 0.99754244, 0.76498276, -0.7068252, 1,
		0.45649233, 0.9993817, 0.8078227, -0.83324534, 1,
		0.45690063, 0.9998353, 0.8369877, -0.91363525, 1,
		0.45703846, 0.9999545, 0.8576527, -0.95835763, 1,
		0.45708582, 0.9999871, 0.8727655, -0.9809275, 1,
		0.45710227, 0.99999624, 0.8840895, -0.99153835, 1,
		0.457108, 0.9999989, 0.89273524, -0.99631274, 1,
		0.45711, 0.9999997, 0.899434, -0.99840873, 1,
		0.4571107, 0.9999999, 0.90468574, -0.9993168, 1,
		0.45711097, 1, 0.9088428, -0.9997075, 1,
		0.45711097, 1, 0.91215944, -0.99987495, 1,
		0.45711097, 1, 0.9148228, -0.9999466, 1,
		0.45711097, 1, 0.9169732, -0.99997723, 1,
		0.45711097, 1, 0.9187171, -0.9999903, 1,
		0.45711097, 1, 0.92013663, -0.9999958, 1,
		0.45711097, 1, 0.9212957, -0.9999982, 1,
		0.45711097, 1, 0.92224455, -0.9999992, 1,
		0.45711097, 1, 0.9230229, -0.9999997, 1,
		0.45711097, 1, 0.9236626, -0.9999999, 1,
		0.45711097, 1, 0.92418903, -0.99999994, 1,
		0.45711097, 1, 0.9246228, -1, 1,
		0.45711097, 1, 0.9249805, -1, 1,
		0.45711097, 1, 0.92527586, -1, 1,
		0.45711097, 1, 0.9255198, -1, 1,
		0.45711097, 1, 0.92572147, -1, 1,
		0.45711097, 1, 0.9258882, -1, 1,
	}
}
