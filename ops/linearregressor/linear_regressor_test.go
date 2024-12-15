package linearregressor

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLinearRegressorInit(t *testing.T) {
	linearRegressor := &LinearRegressor{}
	err := linearRegressor.Init(LinearRegressorOnnxNodeProtoFixture())

	assert.Nil(t, err)
	assert.Equal(t, []float32{1.5, 2.5, 3.5}, linearRegressor.coefficients.Data())
	assert.Equal(t, []float32{0.5}, linearRegressor.intercepts.Data())
	assert.Equal(t, 1, linearRegressor.targets)
}

func TestLinearRegressorInitFailUnsupportedAttribute(t *testing.T) {
	linearRegressor := &LinearRegressor{}
	err := linearRegressor.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "post_transform"}, {Name: "Another"}}})

	expected := ops.ErrUnsupportedAttribute("post_transform", linearRegressor)
	assert.Equal(t, expected, err)
}

func TestLinearRegressorInitFailInvalidAttribute(t *testing.T) {
	linearRegressor := &LinearRegressor{}
	err := linearRegressor.Init(&onnx.NodeProto{Attribute: []*onnx.AttributeProto{{Name: "much_invalid"}}})

	expected := ops.ErrInvalidAttribute("much_invalid", linearRegressor)
	assert.Equal(t, expected, err)
}

func TestLinearRegressor(t *testing.T) {
	tests := []struct {
		version         int64
		attrs           []*onnx.AttributeProto
		shape           []int
		backing         []float32
		expectedShape   tensor.Shape
		expectedBacking []float32
		description     string
	}{
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{-0.45977323}},
				{Name: "intercepts", Floats: []float32{0.21509616}},
				{Name: "targets", I: 1},
			},
			[]int{1, 1},
			[]float32{0.7777024},
			[]int{1, 1},
			[]float32{-0.14247058},
			"linear regressor with 1 input and 1 output variable, 1 sample",
		},
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{-0.45977323}},
				{Name: "intercepts", Floats: []float32{0.21509616}},
				{Name: "targets", I: 1},
			},
			[]int{5, 1},
			[]float32{0.7777024, 0.23754121, 0.82427853, 0.9657492, 0.9726011},
			[]int{5, 1},
			[]float32{-0.14247058, 0.105881065, -0.16388504, -0.22892947, -0.23207982},
			"linear regressor with 1 input and 1 output variable, 5 samples",
		},
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{0.24118852, 0.22617804, 0.27858477}},
				{Name: "intercepts", Floats: []float32{-0.43156273}},
				{Name: "targets", I: 1},
			},
			[]int{1, 3},
			[]float32{0.7777024, 0.23754121, 0.82427853},
			[]int{1, 1},
			[]float32{0.039368242},
			"linear regressor with 3 inputs and 1 output variable, 1 sample",
		},
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{0.24118852, 0.22617804, 0.27858477}},
				{Name: "intercepts", Floats: []float32{-0.43156273}},
				{Name: "targets", I: 1},
			},
			[]int{2, 3},
			[]float32{0.7777024, 0.23754121, 0.82427853, 0.9657492, 0.9726011, 0.45344925},
			[]int{2, 1},
			[]float32{0.039368242, 0.14766997},
			"linear regressor with 3 inputs and 1 output variable, 2 samples",
		},
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{
					0.5384742, 0.36729308, 0.13292366, -0.03843413,
					0.28054297, -0.27832435, 0.4381632, 0.00726224,
					-0.64418418, -0.35812317, 0.69767598, 0.12989015,
				}},
				{Name: "intercepts", Floats: []float32{-0.37036705, -0.34072968, 0.05487297}},
				{Name: "targets", I: 3},
			},
			[]int{1, 4},
			[]float32{0.7777024, 0.23754121, 0.82427853, 0.9657492},
			[]int{1, 3},
			[]float32{0.20810121, 0.17951778, 0.16934107},
			"linear regressor with 4 input and 3 output variables, 1 samples",
		},
		{
			1,
			[]*onnx.AttributeProto{
				{Name: "coefficients", Floats: []float32{
					0.5384742, 0.36729308, 0.13292366, -0.03843413,
					0.28054297, -0.27832435, 0.4381632, 0.00726224,
					-0.64418418, -0.35812317, 0.69767598, 0.12989015,
				}},
				{Name: "intercepts", Floats: []float32{-0.37036705, -0.34072968, 0.05487297}},
				{Name: "targets", I: 3},
			},
			[]int{2, 4},
			[]float32{0.7777024, 0.23754121, 0.82427853, 0.9657492, 0.9726011, 0.45344925, 0.60904247, 0.7755265},
			[]int{2, 3},
			[]float32{0.20810121, 0.17951778, 0.16934107, 0.37105185, 0.0784128, -0.20840444},
			"linear regressor with 4 input and 3 output variables, 2 samples",
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		linearRegressor := linearRegressorVersions[test.version]()
		err := linearRegressor.Init(&onnx.NodeProto{Attribute: test.attrs})
		assert.Nil(t, err, test.description)

		res, err := linearRegressor.Apply(inputs)
		assert.Nil(t, err, test.description)
		assert.Equal(t, test.expectedShape, res[0].Shape(), test.description)
		assert.Equal(t, test.expectedBacking, res[0].Data(), test.description)
	}
}

func TestInputValidationLinearRegressor(t *testing.T) {
	tests := []struct {
		version int64
		inputs  []tensor.Tensor
		err     error
	}{
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int32{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int64{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float32{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]float64{1, 2}, 2)},
			nil,
		},
		{
			1,
			[]tensor.Tensor{},
			ops.ErrInvalidInputCount(0, linearRegressor1BaseOpFixture()),
		},
		{
			1,
			[]tensor.Tensor{ops.TensorWithBackingFixture([]int{1, 2}, 2)},
			ops.ErrInvalidInputType(0, "int", linearRegressor1BaseOpFixture()),
		},
	}

	for _, test := range tests {
		linearRegressor := linearRegressorVersions[test.version]()
		validated, err := linearRegressor.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)

		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func LinearRegressorOnnxNodeProtoFixture() *onnx.NodeProto {
	return &onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "coefficients", Floats: []float32{1.5, 2.5, 3.5}},
			{Name: "intercepts", Floats: []float32{0.5}},
			{Name: "targets", I: 1},
		},
	}
}

func linearRegressor1BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(
		1,
		1,
		1,
		linearRegressorTypeConstraints,
		"linearregressor",
	)
}
