package opset13

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func makeAxisProto(n int) *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{{Name: "axis", I: int64(n)}},
	}
}

func TestGatherInit(t *testing.T) {
	attrs := makeAxisProto(1)
	op := Gather{}
	err := op.Init(attrs)
	assert.NoError(t, err)
	assert.Equal(t, op.axis, 1)
}

func TestGatherInitDefault(t *testing.T) {
	op := Gather{}
	err := op.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{}})
	assert.NoError(t, err)
	assert.Equal(t, op.axis, 0)
}

func TestGatherInitTooManyAttrs(t *testing.T) {
	op := Gather{}
	err := op.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis"}, {Name: "default"}}})
	assert.EqualError(t, err, "gather operator attribute error: invalid count 2 expected 1")
}

func TestGatherInitInvalidAttrName(t *testing.T) {
	op := Gather{}
	err := op.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axes"}}}) // should be axis
	assert.EqualError(t, err, "gather operator attribute error: invalid attribute axes")
}

func TestGather(t *testing.T) {
	tests := []struct {
		data  interface{}
		shape []int

		indices  interface{}
		indShape []int
		axis     int

		expected      interface{}
		expectedShape tensor.Shape
	}{
		// Tip: use numpy's np.take function to make these test cases!
		// i.e. for the second test case:
		// >>> x = np.arange(1, 5)
		// >>> x.shape = (2, 2)
		// >>> i = [0]
		// >>> np.take(x, i, axis=0)
		// Out: array([[1, 2]])
		// >>> np.take(x, i, axis=0).shape
		// Out: (1, 2)

		{
			[]float32{1, 2, 3, 4},
			[]int{4},
			[]int64{0},
			[]int{1},
			0,
			[]float32{1},
			tensor.Shape([]int{1}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{0},
			[]int{1},
			0,
			[]float32{1, 2},
			tensor.Shape([]int{1, 2}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{0},
			[]int{1},
			1,
			[]float32{1, 3},
			tensor.Shape([]int{2, 1}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{0},
			[]int{1},
			-1,
			[]float32{1, 3},
			tensor.Shape([]int{2, 1}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{1},
			[]int{1},
			1,
			[]float32{2, 4},
			tensor.Shape([]int{2, 1}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{0},
			[]int{1, 1},
			1,
			[]float32{1, 3},
			tensor.Shape([]int{2, 1, 1}),
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			[]int{3, 2, 2},
			[]int64{0},
			[]int{1},
			2,
			[]float32{1, 3, 5, 7, 9, 11},
			tensor.Shape([]int{3, 2, 1}),
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			[]int{3, 2, 2},
			[]int64{0},
			[]int{1},
			1,
			[]float32{1, 2, 5, 6, 9, 10},
			tensor.Shape([]int{3, 1, 2}),
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			[]int{3, 3},
			[]int64{0, 2},
			[]int{1, 2},
			1,
			[]float32{1, 3, 4, 6, 7, 9},
			tensor.Shape([]int{3, 1, 2}),
		},

		{
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			[]int{3, 2, 2},
			[]int64{-2},
			[]int{1},
			1,
			[]float32{1, 2, 5, 6, 9, 10},
			tensor.Shape([]int{3, 1, 2}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{4},
			[]int64{-4},
			[]int{1},
			0,
			[]float32{1},
			tensor.Shape([]int{1}),
		},

		{
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int64{0},
			[]int{1},
			-1,
			[]float32{1, 3},
			tensor.Shape([]int{2, 1}),
		},
	}

	for _, test := range tests {
		op := &Gather{test.axis}

		indices := test.indices
		data := test.data

		dataIn := tensor.New(tensor.WithBacking(data), tensor.WithShape(test.shape...))
		indicesIn := tensor.New(tensor.WithBacking(indices), tensor.WithShape(test.indShape...))

		res, err := op.Apply([]tensor.Tensor{dataIn, indicesIn})
		assert.NoError(t, err)
		assert.Equal(t, test.expectedShape, res[0].Shape())
		assert.Equal(t, test.expected, res[0].Data())
	}
}

func TestCombinedWithOtherOp(t *testing.T) {
	concat := &Concat{}
	err := concat.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "axis", I: 0}}})
	assert.NoError(t, err)

	data0 := tensor.New(tensor.WithBacking([]int64{1}), tensor.WithShape(1))
	data1 := tensor.New(tensor.WithBacking([]int64{2}), tensor.WithShape(1))

	data, err := concat.Apply([]tensor.Tensor{data0, data1})
	assert.NoError(t, err)

	gather := &Gather{0}
	indices := tensor.New(tensor.WithBacking([]int64{1}), tensor.WithShape(1))

	res, err := gather.Apply([]tensor.Tensor{data[0], indices})
	assert.NoError(t, err)
	assert.Equal(t, []int64{2}, res[0].Data())
}

func TestScalarInput(t *testing.T) {
	op := &Gather{0}

	dataIn := tensor.New(tensor.WithBacking([]int64{1}), tensor.WithShape(1))

	// Indices is a scalar tensor input.
	indicesIn := tensor.New(tensor.FromScalar(int64(0)))

	res, err := op.Apply([]tensor.Tensor{dataIn, indicesIn})
	assert.NoError(t, err)
	assert.Equal(t, int64(1), res[0].Data())
}

func TestGatherAxesIndexOutOfRange(t *testing.T) {
	op := &Gather{}
	err := op.Init(makeAxisProto(1))
	assert.NoError(t, err)

	dataIn := tensor.New(tensor.WithBacking([]int64{1}), tensor.WithShape(1))
	indicesIn := tensor.New(tensor.WithBacking([]int64{0}), tensor.WithShape(1))

	_, err = op.Apply([]tensor.Tensor{dataIn, indicesIn})
	assert.Error(t, err)
	assert.EqualError(t, err, "axis out of range: axis argument must be in the range -1 <= x < 1, was 1")
}

func TestGatherIndexOutOfRange(t *testing.T) {
	op := &Gather{0}

	dataIn := tensor.New(tensor.WithBacking([]int64{1}), tensor.WithShape(1))
	indicesIn := tensor.New(tensor.WithBacking([]int64{2}), tensor.WithShape(1))

	_, err := op.Apply([]tensor.Tensor{dataIn, indicesIn})
	assert.Error(t, err)
	assert.EqualError(t, err, "axis out of range: all indices entries must be in the range -1 <= x < 1")
}

func TestInputValidationGather(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int32{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]uint32{1, 2}, 2),
				ops.TensorWithBackingFixture([]int64{3, 4}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputCount(1, &Gather{}),
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float64{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{3, 4}, 2),
			},
			ops.ErrInvalidInputType(1, "float32", &Gather{}),
		},
	}

	for _, test := range tests {
		gather := &Gather{}
		validated, err := gather.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}
