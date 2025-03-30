package trilu

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTriluInit(t *testing.T) {
	// Test with upper = 1 (default, upper triangular)
	attrs := makeUpperAttrProto(1)
	op := Trilu{}
	err := op.Init(attrs)
	assert.NoError(t, err)
	assert.Equal(t, true, op.upper)

	// Test with upper = 0 (lower triangular)
	attrs = makeUpperAttrProto(0)
	op = Trilu{}
	err = op.Init(attrs)
	assert.NoError(t, err)
	assert.Equal(t, false, op.upper)
}

func TestTriluInitDefault(t *testing.T) {
	op, ok := newTrilu(14, triluTypeConstraints).(*Trilu)
	assert.True(t, ok)

	err := op.Init(ops.EmptyNodeProto())
	assert.Nil(t, err)
	assert.Equal(t, true, op.upper) // Default is true
}

func TestTriluInitInvalidAttrName(t *testing.T) {
	op := Trilu{BaseOperator: ops.NewBaseOperator(14, 1, 2, triluTypeConstraints, "trilu")}
	err := op.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "invalid"}}})
	assert.EqualError(t, err, "trilu v14 attribute error: invalid attribute invalid")
}

func TestTrilu(t *testing.T) {
	tests := []struct {
		version   int64
		attrs     *onnx.NodeProto
		data      interface{}
		dataShape []int
		k         int64
		expected  interface{}
	}{
		{
			14,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "upper", I: 1},
				},
			},
			[]int64{0, 1, 2, 3, 4, 5, 6, 7, 8},
			[]int{3, 3},
			0,
			[]int{0, 1, 2, 0, 4, 5, 0, 0, 8},
		},
		{
			14,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "upper", I: 0},
				},
			},
			[]int64{0, 1, 2, 3, 4, 5, 6, 7, 8},
			[]int{3, 3},
			0,
			[]int{0, 0, 0, 3, 0, 0, 6, 7, 0},
		},
		{
			14,
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "upper", I: 1},
				},
			},
			[]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
			[]int{3, 4},
			0,
			[]int{0, 1, 2, 0, 4, 5, 0, 0, 8, 0, 0, 11},
		},
	}

	for _, test := range tests {
		op := triluVersions[test.version]()
		err := op.Init(test.attrs)
		assert.Nil(t, err)

		in := ops.TensorWithBackingFixture(test.data, test.dataShape...)

		k := tensor.New(tensor.FromScalar(test.k))

		res, err := op.Apply([]tensor.Tensor{in, k})
		assert.Nil(t, err)

		if err != nil {
			assert.Equal(t, test.expected, res[0].Data())
		}
	}
}

func TestInputValidationTrilu(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			14,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				tensor.New(tensor.FromScalar(int64(0))),
			},
			nil,
		},
		{
			14,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{1, 2}, 2),
			},
			ops.ErrInvalidOptionalInputCount(3, trilu14BaseOpFixture()),
		},
		{
			14,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
				tensor.New(tensor.FromScalar(0)),
			},
			ops.ErrInvalidInputType(0, "int", trilu14BaseOpFixture()),
		},
	}

	for _, test := range tests {
		trilu := triluVersions[test.version]()
		validated, err := trilu.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func trilu14BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(14, 1, 2, triluTypeConstraints, "trilu")
}

func makeUpperAttrProto(upper int64) *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{{Name: "upper", I: upper}},
	}
}
