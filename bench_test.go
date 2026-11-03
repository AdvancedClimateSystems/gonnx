package gonnx

import (
	"testing"
)

func BenchmarkModels(b *testing.B) {
	benchmarks := []struct {
		name   string
		path   string
		inputs Tensors
	}{
		{
			"scaler",
			"./sample_models/onnx_models/scaler.onnx",
			tensorsFixture([]string{"X"}, [][]int{{2, 3}}, [][]float32{rangeFloat(6)}),
		},
		{
			"mlp",
			"./sample_models/onnx_models/mlp.onnx",
			tensorsFixture([]string{"data_input"}, [][]int{{2, 3}}, [][]float32{rangeFloat(6)}),
		},
		{
			"gru",
			"./sample_models/onnx_models/gru.onnx",
			tensorsFixture(
				[]string{"data_input", "init_hidden"},
				[][]int{{1, 30, 3}, {1, 1, 5}},
				[][]float32{rangeFloat(90), make([]float32, 5)},
			),
		},
		{
			"ndm",
			"./sample_models/onnx_models/ndm.onnx",
			tensorsFixture(
				[]string{"sensor_input", "setpoint_input"},
				[][]int{{1, 18, 4}, {1, 1}},
				[][]float32{rangeFloat(72), rangeFloat(1)},
			),
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			model, err := NewModelFromFile(bm.path)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				if _, err := model.Run(bm.inputs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
