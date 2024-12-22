package cumsum

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCumSumInit(t *testing.T) {
	c := &CumSum{}
	err := c.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "exclusive", I: 1},
				{Name: "reverse", I: 1},
			},
		},
	)

	assert.Nil(t, err)
	assert.Equal(t, true, c.exclusive)
	assert.Equal(t, true, c.reverse)
}

func TestCumSumInitDefaults(t *testing.T) {
	c := &CumSum{}
	err := c.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{},
		},
	)

	assert.Nil(t, err)
	assert.Equal(t, false, c.exclusive)
	assert.Equal(t, false, c.reverse)
}

func TestCumSum(t *testing.T) {
	tests := []struct {
		version  int64
		node     *onnx.NodeProto
		backing  []float32
		axis     int32
		shape    []int
		expected []float32
	}{
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "exclusive", I: 0},
					{Name: "reverse", I: 0},
				},
			},
			[]float32{1, 2, 3, 4},
			0,
			[]int{2, 2},
			[]float32{1, 2, 4, 6},
		},
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "exclusive", I: 0},
					{Name: "reverse", I: 0},
				},
			},
			[]float32{1, 2, 3, 4},
			1,
			[]int{2, 2},
			[]float32{1, 3, 3, 7},
		},
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "exclusive", I: 1},
					{Name: "reverse", I: 0},
				},
			},
			[]float32{1, 2, 3},
			0,
			[]int{3},
			[]float32{0, 1, 3},
		},
		{
			11,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "exclusive", I: 0},
					{Name: "reverse", I: 1},
				},
			},
			[]float32{1, 2, 3},
			0,
			[]int{3},
			[]float32{6, 5, 3},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
			tensor.New(tensor.FromScalar(test.axis)),
		}

		cumsum := cumsumVersions[test.version]()
		err := cumsum.Init(test.node)
		assert.Nil(t, err)

		res, err := cumsum.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}
