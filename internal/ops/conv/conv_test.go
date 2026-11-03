package conv

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
	"github.com/stretchr/testify/assert"
)

func TestConvInit(t *testing.T) {
	c := &Conv{}
	err := c.Init(Conv2DOnnxNodeProtoFixture())

	assert.Nil(t, err)

	var expectedAutopad AutoPadSetting = "VALID"

	assert.Equal(t, expectedAutopad, c.autoPad)
	assert.Equal(t, []int{1, 1}, c.dilations)
	assert.Equal(t, []int{2, 2}, c.kernelShape)
	assert.Equal(t, []int{1, 2}, c.pads)
	assert.Equal(t, []int{1, 1}, c.strides)
}

func TestConvInitUnsupported(t *testing.T) {
	c := &Conv{}
	err := c.Init(ConvUnsupportedOnnxNodeProtoFixture())

	assert.Equal(
		t,
		err,
		ops.ErrUnsupportedAttribute("group", c),
	)
}

func TestConv(t *testing.T) {
	tests := []struct {
		version       int64
		node          *onnx.NodeProto
		shapes        [][]int
		backings      [][]float32
		expectedShape tensor.Shape
		expected      []float32
	}{
		// Test 1D Convolution.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{3}},
					{Name: "pads", Ints: []int64{0, 0}},
					{Name: "strides", Ints: []int64{1}},
				},
			},
			[][]int{{1, 1, 6}, {1, 1, 3}},
			[][]float32{{0, 1, 2, 3, 4, 5}, {1, 1, 1}},
			[]int{1, 1, 4},
			[]float32{3, 6, 9, 12},
		},
		// Test 2D Convolution.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}},
			[]int{1, 1, 2, 2},
			[]float32{8, 12, 20, 24},
		},
		// Test SAME_LOWER autopad setting.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("SAME_LOWER")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}},
			[]int{1, 1, 3, 3},
			[]float32{0, 1, 3, 3, 8, 12, 9, 20, 24},
		},
		// Test SAME_UPPER autopad setting.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("SAME_UPPER")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}},
			[]int{1, 1, 3, 3},
			[]float32{8, 12, 7, 20, 24, 13, 13, 15, 8},
		},
		// Test VALID autopad setting.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("VALID")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}},
			[]int{1, 1, 2, 2},
			[]float32{8, 12, 20, 24},
		},
		// Test dilation attribute.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{2, 2}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 4, 4}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 1, 1, 1}},
			[]int{1, 1, 2, 2},
			[]float32{20, 24, 36, 40},
		},
		// Test pads attribute.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{1, 1}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{1, 1, 2, 2}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 2, 2}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3}, {1, 1, 1, 1}},
			[]int{1, 1, 4, 4},
			[]float32{0, 1, 1, 0, 2, 6, 4, 0, 2, 5, 3, 0, 0, 0, 0, 0},
		},
		// Test strides attribute.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{2, 2}},
				},
			},
			[][]int{{1, 1, 4, 4}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 1, 1, 1}},
			[]int{1, 1, 2, 2},
			[]float32{10, 18, 42, 50},
		},
		// Test batch dimension.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{2, 1, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, {1, 1, 1, 1}},
			[]int{2, 1, 2, 2},
			[]float32{8, 12, 20, 24, 44, 48, 56, 60},
		},
		// Test 2D convolution with multiple channels.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 2, 3, 3}, {1, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, {1, 1, 1, 1}},
			[]int{1, 1, 2, 2},
			[]float32{52, 60, 76, 84},
		},
		// Test multiple kernels.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {2, 1, 2, 2}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1, 2, 2, 2, 2}},
			[]int{1, 2, 2, 2},
			[]float32{8, 12, 20, 24, 16, 24, 40, 48},
		},
		// Test bias.
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "auto_pad", S: []byte("NOTSET")},
					{Name: "dilations", Ints: []int64{}},
					{Name: "group", I: 1},
					{Name: "kernel_shape", Ints: []int64{2, 2}},
					{Name: "pads", Ints: []int64{0, 0, 0, 0}},
					{Name: "strides", Ints: []int64{1, 1}},
				},
			},
			[][]int{{1, 1, 3, 3}, {1, 1, 2, 2}, {1}},
			[][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}, {0.5}},
			[]int{1, 1, 2, 2},
			[]float32{8.5, 12.5, 20.5, 24.5},
		},
	}

	for _, test := range tests {
		inputs := []*tensor.Tensor{
			ops.TensorWithBackingFixture(test.backings[0], test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backings[1], test.shapes[1]...),
			nil,
		}

		if len(test.backings) == 3 {
			inputs[2] = ops.TensorWithBackingFixture(test.backings[2], test.shapes[2]...)
		}

		conv := convVersions[test.version]()
		err := conv.Init(test.node)
		assert.Nil(t, err)

		res, err := conv.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestInputValidationConv(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []*tensor.Tensor
		err     error
	}{
		{
			11,
			[]*tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
				nil,
			},
			nil,
		},
		{
			11,
			[]*tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float64{3, 4}, 2),
				ops.TensorWithBackingFixture([]float64{5, 6}, 2),
			},
			nil,
		},
		{
			11,
			[]*tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{true, true}, 2),
			},
			ops.ErrInvalidOptionalInputCount(1, conv11BaseOpFixture()),
		},
		{
			11,
			[]*tensor.Tensor{
				ops.TensorWithBackingFixture([]bool{true, true}, 2),
				ops.TensorWithBackingFixture([]bool{true, true}, 2),
			},
			ops.ErrInvalidInputType(0, "bool", conv11BaseOpFixture()),
		},
	}

	for _, test := range tests {
		conv := convVersions[test.version]()
		validated, err := conv.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func TestConvParams(t *testing.T) {
	tests := []struct {
		conv        *Conv
		xShape      tensor.Shape
		kernelShape tensor.Shape
		expected    convParams
	}{
		{
			&Conv{autoPad: NotSet},
			tensor.Shape{1, 1, 3, 3},
			tensor.Shape{1, 1, 2, 2},
			convParams{kernel: [2]int{2, 2}, strides: [2]int{1, 1}, dilations: [2]int{1, 1}},
		},
		{
			&Conv{autoPad: NotSet, strides: []int{2, 3}, dilations: []int{2, 1}, pads: []int{1, 2, 3, 4}},
			tensor.Shape{1, 1, 6, 6},
			tensor.Shape{1, 1, 2, 2},
			convParams{
				kernel: [2]int{2, 2}, strides: [2]int{2, 3}, dilations: [2]int{2, 1},
				padsBegin: [2]int{1, 2}, padsEnd: [2]int{3, 4},
			},
		},
		{
			&Conv{autoPad: SameLower},
			tensor.Shape{1, 1, 3, 3},
			tensor.Shape{1, 1, 2, 2},
			convParams{
				kernel: [2]int{2, 2}, strides: [2]int{1, 1}, dilations: [2]int{1, 1},
				padsBegin: [2]int{1, 1},
			},
		},
		{
			&Conv{autoPad: SameUpper},
			tensor.Shape{1, 1, 3, 3},
			tensor.Shape{1, 1, 2, 2},
			convParams{
				kernel: [2]int{2, 2}, strides: [2]int{1, 1}, dilations: [2]int{1, 1},
				padsEnd: [2]int{1, 1},
			},
		},
		{
			&Conv{autoPad: Valid, pads: []int{1, 1, 1, 1}},
			tensor.Shape{1, 1, 3, 3},
			tensor.Shape{1, 1, 2, 2},
			convParams{kernel: [2]int{2, 2}, strides: [2]int{1, 1}, dilations: [2]int{1, 1}},
		},
		{
			&Conv{autoPad: NotSet, strides: []int{2}, pads: []int{1, 0}},
			tensor.Shape{1, 1, 6},
			tensor.Shape{1, 1, 3},
			convParams{
				kernel: [2]int{1, 3}, strides: [2]int{1, 2}, dilations: [2]int{1, 1},
				padsBegin: [2]int{0, 1},
			},
		},
	}

	for _, test := range tests {
		params, err := test.conv.convParams(test.xShape, test.kernelShape)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, params)
	}
}

func TestConvParamsInvalidKernelShape(t *testing.T) {
	conv := &Conv{kernelShape: []int{3, 3}}

	_, err := conv.convParams(tensor.Shape{1, 1, 3, 3}, tensor.Shape{1, 1, 2, 2})

	assert.Error(t, err)
}

func Conv2DOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "auto_pad", S: []byte("VALID")},
			{Name: "dilations", Ints: []int64{1, 1}},
			{Name: "kernel_shape", Ints: []int64{2, 2}},
			{Name: "pads", Ints: []int64{1, 2}},
			{Name: "strides", Ints: []int64{1, 1}},
		},
	}
}

func ConvUnsupportedOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "group", I: 2},
		},
	}
}

func conv11BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(11, 2, 3, convTypeConstraints, "conv")
}
