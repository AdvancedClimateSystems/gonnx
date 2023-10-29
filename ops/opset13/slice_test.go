package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSliceInit(t *testing.T) {
	s := &Slice{}

	// since the slice does not have any attributes we pass in nil. This should not
	// fail initializing the slice.
	err := s.Init(nil)
	assert.Nil(t, err)
}

func TestSlice(t *testing.T) {
	tests := []struct {
		shape           []int
		starts          []int64
		ends            []int64
		axes            []int64
		steps           []int64
		expectedShape   tensor.Shape
		expectedBacking []float32
	}{
		{
			[]int{2, 3},
			[]int64{1, 0},
			[]int64{2, 2},
			nil,
			nil,
			[]int{2},
			[]float32{3, 4},
		},
		{
			[]int{3, 3},
			[]int64{1},
			[]int64{3},
			[]int64{0},
			nil,
			[]int{2, 3},
			[]float32{3, 4, 5, 6, 7, 8},
		},
		{
			[]int{3, 3},
			[]int64{1},
			[]int64{3},
			[]int64{1},
			nil,
			[]int{3, 2},
			[]float32{1, 2, 4, 5, 7, 8},
		},
		{
			[]int{2, 3, 3},
			[]int64{0, 1, 1},
			[]int64{1, 3, 3},
			nil,
			nil,
			[]int{2, 2},
			[]float32{4, 5, 7, 8},
		},
		{
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
		slice := &Slice{}
		inputs := []tensor.Tensor{
			ops.Float32TensorFixture(test.shape...),
			ops.TensorWithBackingFixture(test.starts, len(test.starts)),
			ops.TensorWithBackingFixture(test.ends, len(test.ends)),
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
		slice          *Slice
		starts         []int
		ends           []int
		axes           []int
		steps          []int
		nSlices        int
		expectedSlices []tensor.Slice
	}{
		{
			&Slice{},
			[]int{1, 0},
			[]int{2, 3},
			[]int{0, 1},
			[]int{1, 1},
			2,
			[]tensor.Slice{ops.NewSlicer(1, 2, 1), ops.NewSlicer(0, 3, 1)},
		},
		{
			&Slice{},
			[]int{0, 2},
			[]int{2, 5},
			[]int{2, 0},
			[]int{1, 2},
			3,
			[]tensor.Slice{ops.NewSlicer(2, 5, 2), nil, ops.NewSlicer(0, 2, 1)},
		},
	}

	for _, test := range tests {
		slices := test.slice.constructSlices(
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
	slice := &Slice{}
	res := slice.getDefaultAxes(3)
	assert.Equal(t, []int{0, 1, 2}, res)
}

func TestGetDefaultSteps(t *testing.T) {
	slice := &Slice{}
	res := slice.getDefaultSteps(3)
	assert.Equal(t, []int{1, 1, 1}, res)
}

func TestInputValidationSlice(t *testing.T) {
	tests := []struct {
		inputs   []tensor.Tensor
		expected []tensor.Tensor
		err      error
	}{
		{
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
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			nil,
			fmt.Errorf("slice operator: expected 3-5 input tensors, got 1"),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
				ops.TensorWithBackingFixture([]int{3, 4}, 2),
			},
			nil,
			fmt.Errorf("slice operator: input 1 does not allow type int"),
		},
	}

	for _, test := range tests {
		slice := &Slice{}
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
