package slice

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSliceInit(t *testing.T) {
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
					{Name: "starts", Ints: []int64{1, 0}},
					{Name: "ends", Ints: []int64{2, 2}},
				},
			},
			nil,
		},
		{
			10,
			nil,
			nil,
		},
		{
			11,
			nil,
			nil,
		},
		{
			13,
			nil,
			nil,
		},
	}

	for _, test := range tests {
		op := sliceVersions[test.version]()
		err := op.Init(test.attrs)
		assert.Equal(t, test.err, err)
	}
}

func TestSlice(t *testing.T) {
	tests := []struct {
		version int64
		attrs   *onnx.NodeProto

		shape           []int
		starts          []int64
		ends            []int64
		axes            []int64
		steps           []int64
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			1,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0, 1}},
					{Name: "starts", Ints: []int64{1, 0}},
					{Name: "ends", Ints: []int64{2, 3}},
				},
			},
			[]int{2, 4},
			nil,
			nil,
			nil,
			nil,
			[]int{3},
			[]float32{4, 5, 6},
		},
		{
			13,
			nil,
			[]int{2, 3},
			[]int64{1, 0},
			[]int64{2, 2},
			nil,
			nil,
			[]int{2},
			[]float32{3, 4},
		},
		{
			13,
			nil,
			[]int{2, 3},
			[]int64{1, 0},
			[]int64{2, 2},
			nil,
			nil,
			[]int{2},
			[]float32{3, 4},
		},
		{
			13,
			nil,
			[]int{3, 3},
			[]int64{1},
			[]int64{3},
			[]int64{0},
			nil,
			[]int{2, 3},
			[]float32{3, 4, 5, 6, 7, 8},
		},
		{
			13,
			nil,
			[]int{3, 3},
			[]int64{1},
			[]int64{3},
			[]int64{1},
			nil,
			[]int{3, 2},
			[]float32{1, 2, 4, 5, 7, 8},
		},
		{
			13,
			nil,
			[]int{2, 3, 3},
			[]int64{0, 1, 1},
			[]int64{1, 3, 3},
			nil,
			nil,
			[]int{2, 2},
			[]float32{4, 5, 7, 8},
		},
		{
			13,
			nil,
			[]int{4, 4},
			[]int64{0},
			[]int64{4},
			nil,
			[]int64{2},
			[]int{2, 4},
			[]float32{0, 1, 2, 3, 8, 9, 10, 11},
		},
	}

	for _, test := range tests {
		slice := sliceVersions[test.version]()
		err := slice.Init(test.attrs)
		assert.Nil(t, err)

		var inputs []tensor.Tensor
		if test.version >= 10 {
			inputs = []tensor.Tensor{
				ops.Float32TensorFixture(test.shape...),
				ops.TensorWithBackingFixture(test.starts, len(test.starts)),
				ops.TensorWithBackingFixture(test.ends, len(test.ends)),
			}
		} else {
			inputs = []tensor.Tensor{
				ops.Float32TensorFixture(test.shape...),
			}
		}

		if test.axes != nil {
			axesNode := ops.TensorWithBackingFixture(test.axes, len(test.axes))
			inputs = append(inputs, axesNode)
		} else {
			inputs = append(inputs, nil)
		}

		if test.steps != nil {
			stepNode := ops.TensorWithBackingFixture(test.steps, len(test.steps))
			inputs = append(inputs, stepNode)
		} else {
			inputs = append(inputs, nil)
		}

		res, err := slice.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func TestConstructSlices(t *testing.T) {
	tests := []struct {
		starts         []int
		ends           []int
		axes           []int
		steps          []int
		nSlices        int
		expectedSlices []tensor.Slice
	}{
		{
			[]int{1, 0},
			[]int{2, 3},
			[]int{0, 1},
			[]int{1, 1},
			2,
			[]tensor.Slice{ops.NewSlicer(1, 2, 1), ops.NewSlicer(0, 3, 1)},
		},
		{
			[]int{0, 2},
			[]int{2, 5},
			[]int{2, 0},
			[]int{1, 2},
			3,
			[]tensor.Slice{ops.NewSlicer(2, 5, 2), nil, ops.NewSlicer(0, 2, 1)},
		},
	}

	for _, test := range tests {
		slices := constructSlices(
			test.starts, test.ends, test.steps, test.axes, test.nSlices,
		)

		assert.Equal(t, test.nSlices, len(slices))

		for i := 0; i < test.nSlices; i++ {
			if test.expectedSlices[i] == nil {
				assert.Nil(t, slices[i])
			} else {
				assert.Equal(t, test.expectedSlices[i].Start(), slices[i].Start())
				assert.Equal(t, test.expectedSlices[i].End(), slices[i].End())
				assert.Equal(t, test.expectedSlices[i].Step(), slices[i].Step())
			}
		}
	}
}

func TestGetDefaultAxes(t *testing.T) {
	res := getDefaultAxes(3)
	assert.Equal(t, []int{0, 1, 2}, res)
}

func TestGetDefaultSteps(t *testing.T) {
	res := getDefaultSteps(3)
	assert.Equal(t, []int{1, 1, 1}, res)
}

func TestInputValidationSlice(t *testing.T) {
	tests := []struct {
		version  int64
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
			nil,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
				nil,
				nil,
			},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, slice1BaseOpFixture()),
		},
		{
			10,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, slice10BaseOpFixture()),
		},
		{
			11,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, slice11BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			ops.ErrInvalidOptionalInputCount(1, slice13BaseOpFixture()),
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			ops.ErrInvalidInputType(1, "int", slice13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		slice := sliceVersions[test.version]()
		validated, err := slice.ValidateInputs(test.inputs)

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

func slice1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(1, 3, 5, sliceTypeConstraints, "slice")
}

func slice10BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(10, 3, 5, sliceTypeConstraints, "slice")
}

func slice11BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(11, 3, 5, sliceTypeConstraints, "slice")
}

func slice13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 3, 5, sliceTypeConstraints, "slice")
}
