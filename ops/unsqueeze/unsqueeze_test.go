package unsqueeze

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestUnsqueezeInit(t *testing.T) {
	tests := []struct {
		version int64
		attrs   *onnx.NodeProto
		err     error
	}{
		{
			1,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1, 0}},
				},
			},
			nil,
		},
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1, 0}},
				},
			},
			nil,
		},
		{
			13,
			nil,
			nil,
		},
	}

	for _, test := range tests {
		op := unsqueezeVersions[test.version]()
		err := op.Init(test.attrs)
		assert.Equal(t, test.err, err)
	}
}

func TestAxesOutRangeError(t *testing.T) {
	op := unsqueezeVersions[13]()
	err := op.Init(nil)
	assert.Nil(t, err)

	axes := []int64{4}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	expected := ops.ErrNotAllAxesInRange(3, 3)
	assert.Equal(t, err, expected)
}

func TestDuplicateEntriesAfterOffsetNotAllowed(t *testing.T) {
	op := unsqueezeVersions[13]()
	err := op.Init(nil)
	assert.Nil(t, err)

	// -1 will be offset to 3 (since outputrank = 4)
	axes := []int64{3, -1}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	assert.EqualError(t, err, "invalid input tensor for unsqueeze v13: axes cannot have duplicate entries after offset")
}

func TestDuplicateEntriesNotAllowed(t *testing.T) {
	op := unsqueezeVersions[13]()
	err := op.Init(nil)
	assert.Nil(t, err)

	axes := []int64{0, 0}
	data := ops.Arange(9, 1) // 3 x 3 tensor

	dataIn := ops.TensorWithBackingFixture(data, 3, 3)
	axesIn := ops.TensorWithBackingFixture(axes, len(axes))
	_, err = op.Apply([]tensor.Tensor{dataIn, axesIn})
	assert.EqualError(t, err, "invalid input tensor for unsqueeze v13: axes cannot have duplicate entries after offset")
}

func TestUnsqueeze(t *testing.T) {
	tests := []struct {
		data              interface{}
		dataShape         []int
		axes              []int64
		expectOutputShape []int
	}{
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{0}, []int{1, 2, 2}},
		{[]int64{1}, []int{1}, []int64{1}, []int{1, 1}},
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{0, -1}, []int{1, 2, 2, 1}},
		{[]int64{1, 2, 3, 4}, []int{2, 2}, []int64{-1, 0}, []int{1, 2, 2, 1}},

		{
			[]int16{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{0, 2, 4, 6},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]complex128{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{6, 0, 4, 2},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{-7, -5, -3, -1},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{-1, -7, -3, -5},
			[]int{1, 2, 1, 2, 1, 2, 1},
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]int{2, 2, 2},
			[]int64{0, 1, 2, 3},
			[]int{1, 1, 1, 1, 2, 2, 2},
		},
	}
	for _, test := range tests {
		op := &Unsqueeze{}
		err := op.Init(nil)
		assert.Nil(t, err)

		axes := test.axes
		data := test.data
		dataIn := ops.TensorWithBackingFixture(data, test.dataShape...)
		axesIn := ops.TensorWithBackingFixture(axes, len(axes))

		res, err := op.Apply([]tensor.Tensor{dataIn, axesIn})
		assert.NoError(t, err)

		shape := res[0].Shape()
		expShape := tensor.Shape(test.expectOutputShape)
		assert.Equal(t, expShape, shape)
	}
}

func TestInputValidationUnsqueeze(t *testing.T) {
	tests := []struct {
		inputs  []tensor.Tensor
		version int64
		err     error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			1,
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			11,
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			13,
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			13,
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			1,
			ops.ErrInvalidInputCount(2, flatten1BaseOpFixture()),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			11,
			ops.ErrInvalidInputCount(2, flatten11BaseOpFixture()),
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			13,
			ops.ErrInvalidInputCount(1, flatten13BaseOpFixture()),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			13,
			ops.ErrInvalidInputType(0, "int", flatten13BaseOpFixture()),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			13,
			ops.ErrInvalidInputType(1, "int32", flatten13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		unsqueeze := unsqueezeVersions[test.version]()
		validated, err := unsqueeze.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func flatten1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 1, 1, [][]tensor.Dtype{ops.AllTypes}, "unsqueeze")
}

func flatten11BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(11, 1, 1, [][]tensor.Dtype{ops.AllTypes}, "unsqueeze")
}

func flatten13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 2, [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}, "unsqueeze")
}
