package reducemean

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReduceMeanInit(t *testing.T) {
	r := &ReduceMean{}
	err := r.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axes", Ints: []int64{1, 3}},
			{Name: "keepdims", I: 0},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, []int{1, 3}, r.axes)
	assert.Equal(t, false, r.keepDims)
}

func TestReduceMean(t *testing.T) {
	tests := []struct {
		version         int64
		node            *onnx.NodeProto
		backing         []float32
		shape           []int
		expectedBacking []float32
		expectedShape   tensor.Shape
	}{
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{1, 2},
			[]int{2},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{1, 2},
			[]int{1, 2},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0.5, 2.5},
			[]int{2},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0.5, 2.5},
			[]int{2, 1},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{1.5, 2.5, 3.5},
			[]int{3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{1.5, 2.5, 3.5},
			[]int{1, 3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{1, 4},
			[]int{2},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5},
			[]int{2, 3},
			[]float32{1, 4},
			[]int{2, 1},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{1.5, 2.5, 3.5, 7.5, 8.5, 9.5},
			[]int{2, 3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{1.5, 2.5, 3.5, 7.5, 8.5, 9.5},
			[]int{2, 1, 3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0, 1}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{4.5, 5.5, 6.5},
			[]int{3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0, 1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{4.5, 5.5, 6.5},
			[]int{1, 1, 3},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1, 2}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{2.5, 8.5},
			[]int{2},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1, 2}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{2, 2, 3},
			[]float32{2.5, 8.5},
			[]int{2, 1, 1},
		},
		{
			13,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{-1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{0.5, 2.5},
			[]int{2, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		reduceMean := reduceMeanVersions[test.version]()
		err := reduceMean.Init(test.node)
		assert.Nil(t, err)

		res, err := reduceMean.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationReduceMean(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint64{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			nil,
		},
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
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			ops.ErrInvalidInputCount(2, reduceMean13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", reduceMean13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		reduceMean := reduceMeanVersions[test.version]()
		validated, err := reduceMean.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func reduceMean13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, reduceMeanTypeConstraints, "reducemean")
}
