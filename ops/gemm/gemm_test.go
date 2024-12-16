package gemm

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGemmInit(t *testing.T) {
	gemm := Gemm{}
	err := gemm.Init(GemmOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, float32(10.0), gemm.alpha)
	assert.Equal(t, float32(0.98), gemm.beta)
	assert.Equal(t, true, gemm.transA)
	assert.Equal(t, true, gemm.transB)
}

func TestGemmInitFail(t *testing.T) {
	gemm := &Gemm{}
	err := gemm.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "unknownAttribute"}}})

	expected := ops.ErrInvalidAttribute("unknownAttribute", gemm)
	assert.Equal(t, expected, err)
}

func TestGemm(t *testing.T) {
	tests := []struct {
		version  int64
		attrs    *onnx.NodeProto
		shapes   [][]int
		expected []float32
	}{
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 0},
					{Name: "transB", I: 0},
				},
			},
			[][]int{{3, 2}, {2, 5}, {5}},
			[]float32{5, 7, 9, 11, 13, 15, 21, 27, 33, 39, 25, 35, 45, 55, 65},
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 1},
					{Name: "transB", I: 0},
				},
			},
			[][]int{{2, 3}, {2, 5}, {5}},
			[]float32{15, 19, 23, 27, 31, 20, 26, 32, 38, 44, 25, 33, 41, 49, 57},
		},
		{
			9,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 1},
					{Name: "transB", I: 1},
				},
			},
			[][]int{{2, 3}, {5, 2}, {5}},
			[]float32{3, 10, 17, 24, 31, 4, 15, 26, 37, 48, 5, 20, 35, 50, 65},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 0},
					{Name: "transB", I: 1},
				},
			},
			[][]int{{3, 2}, {5, 2}, {5}},
			[]float32{1, 4, 7, 10, 13, 3, 14, 25, 36, 47, 5, 24, 43, 62, 81},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 0},
					{Name: "transB", I: 0},
				},
			},
			[][]int{{1, 2}, {2, 5}, {5}},
			[]float32{5, 7, 9, 11, 13},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 0},
					{Name: "transB", I: 0},
				},
			},
			[][]int{{1, 2}, {2, 5}},
			[]float32{5, 6, 7, 8, 9},
		},
		{
			7,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: 1.0},
					{Name: "beta", F: 1.0},
					{Name: "transA", I: 0},
					{Name: "transB", I: 0},
				},
			},
			[][]int{{20, 4}, {4, 6}, {6}},
			[]float32{
				84, 91, 98, 105, 112, 119, 228, 251, 274,
				297, 320, 343, 372, 411, 450, 489, 528, 567, 516, 571,
				626, 681, 736, 791, 660, 731, 802, 873, 944, 1015, 804,
				891, 978, 1065, 1152, 1239, 948, 1051, 1154, 1257, 1360,
				1463, 1092, 1211, 1330, 1449, 1568, 1687, 1236, 1371,
				1506, 1641, 1776, 1911, 1380, 1531, 1682, 1833, 1984,
				2135, 1524, 1691, 1858, 2025, 2192, 2359, 1668, 1851,
				2034, 2217, 2400, 2583, 1812, 2011, 2210, 2409, 2608,
				2807, 1956, 2171, 2386, 2601, 2816, 3031, 2100, 2331,
				2562, 2793, 3024, 3255, 2244, 2491, 2738, 2985, 3232,
				3479, 2388, 2651, 2914, 3177, 3440, 3703, 2532, 2811,
				3090, 3369, 3648, 3927, 2676, 2971, 3266, 3561, 3856,
				4151, 2820, 3131, 3442, 3753, 4064, 4375,
			},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.shapes[0]...),
			ops.Float32TensorFixture(test.shapes[1]...),
		}
		if len(test.shapes) == 3 {
			inputs = append(inputs, ops.Float32TensorFixture(test.shapes[2]...))
		} else {
			inputs = append(inputs, nil)
		}

		gemm := gemmVersions[test.version]()
		err := gemm.Init(test.attrs)
		assert.Nil(t, err)

		res, err := gemm.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationGemm(t *testing.T) {
	tests := []struct {
		version  int64
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
				nil,
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
				ops.TensorWithBackingFixture([]float32{5, 6}, 2),
			},
			nil,
			nil,
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, gemm13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
			ops.ErrInvalidOptionalInputCount(4, gemm13BaseOpFixture()),
		},
		{
			7,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]uint32{3, 4}, 2),
				ops.TensorWithBackingFixture([]uint32{5, 6}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "uint32", gemm7BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			ops.ErrInvalidInputType(0, "int", gemm13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		gemm := gemmVersions[test.version]()
		validated, err := gemm.ValidateInputs(test.inputs)

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

func GemmOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "alpha", F: 10.0},
			{Name: "beta", F: 0.98},
			{Name: "transA", I: 1},
			{Name: "transB", I: 1},
		},
	}
}

func gemm7BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(
		7,
		3,
		3,
		[][]tensor.Dtype{
			{tensor.Float32, tensor.Float64},
			{tensor.Float32, tensor.Float64},
			{tensor.Float32, tensor.Float64},
		},
		"gemm",
	)
}

func gemm13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 3, gemmTypeConstraints, "gemm")
}
