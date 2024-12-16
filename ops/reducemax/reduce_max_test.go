package reducemax

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReduceMaxInit(t *testing.T) {
	tests := []struct {
		version int64
		err     error
	}{
		{1, nil},
		{11, nil},
		{12, nil},
		{13, nil},
	}

	for _, test := range tests {
		r, ok := reduceMaxVersions[test.version]().(*ReduceMax)
		assert.True(t, ok)

		err := r.Init(&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "axes", Ints: []int64{1, 3}},
				{Name: "keepdims", I: 0},
			},
		})

		assert.Equal(t, test.err, err)
		assert.Equal(t, []int{1, 3}, r.axes)
		assert.Equal(t, false, r.keepDims)
	}
}

func TestReduceMax(t *testing.T) {
	tests := []struct {
		version         int64
		attrs           *onnx.NodeProto
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
			[]float32{2, 3},
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
			[]float32{2, 3},
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
			[]float32{1, 3},
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
			[]float32{1, 3},
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
			[]float32{3, 4, 5},
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
			[]float32{3, 4, 5},
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
			[]float32{2, 5},
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
			[]float32{2, 5},
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
			[]float32{3, 4, 5, 9, 10, 11},
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
			[]float32{3, 4, 5, 9, 10, 11},
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
			[]float32{9, 10, 11},
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
			[]float32{9, 10, 11},
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
			[]float32{5, 11},
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
			[]float32{5, 11},
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
			[]float32{1, 3},
			[]int{2, 1},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		reduceMax := reduceMaxVersions[test.version]()
		err := reduceMax.Init(test.attrs)
		assert.Nil(t, err)

		res, err := reduceMax.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestInputValidationReduceMax(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int8{1, 2}, 2),
			},
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint8{1, 2}, 2),
			},
			nil,
		},
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
			ops.ErrInvalidInputCount(2, reduceMax13BaseOpFixture()),
		},
		{
			1,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int8{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int8", reduceMax1BaseOpFixture()),
		},
		{
			11,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint8{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "uint8", reduceMax11BaseOpFixture()),
		},
		{
			12,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", reduceMax12BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", reduceMax13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		reduceMax := reduceMaxVersions[test.version]()
		validated, err := reduceMax.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func reduceMax1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 1, 1, reduceMax11TypeConstraints, "reducemax")
}

func reduceMax11BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(11, 1, 1, reduceMax11TypeConstraints, "reducemax")
}

func reduceMax12BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(12, 1, 1, reduceMaxTypeConstraints, "reducemax")
}

func reduceMax13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 1, reduceMaxTypeConstraints, "reducemax")
}
