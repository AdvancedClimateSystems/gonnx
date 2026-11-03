package linearregressor

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var linearRegressorTypeConstraints = [][]tensor.DataType{
	{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// PostTransformOption describes all possible post transform options for the
// linear regressor operator.
type postTransformOption string

const (
	noTransform          postTransformOption = "NONE"
	softmaxTransform     postTransformOption = "SOFTMAX"
	logisticTransform    postTransformOption = "LOGISTIC"
	softmaxZeroTransform postTransformOption = "SOFTMAX_ZERO"
	probitTransform      postTransformOption = "PROBIT"
)

// LinearRegressor represents the ONNX-ml linearRegressor operator.
type LinearRegressor struct {
	ops.BaseOperator

	coefficients  *tensor.Tensor
	intercepts    *tensor.Tensor
	postTransform postTransformOption
	targets       int
}

// newLinearRegressor creates a new linearRegressor operator.
func newLinearRegressor(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &LinearRegressor{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"linearregressor",
		),
		postTransform: noTransform,
		targets:       1,
	}
}

// Init initializes the linearRegressor operator.
func (l *LinearRegressor) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "coefficients":
			floats := attr.GetFloats()
			l.coefficients = tensor.New(floats, len(floats))
		case "intercepts":
			floats := attr.GetFloats()
			l.intercepts = tensor.New(floats, len(floats))
		case "post_transform":
			return ops.ErrUnsupportedAttribute(attr.GetName(), l)
		case "targets":
			l.targets = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	coefficients, err := l.coefficients.Reshape(l.targets, l.coefficients.Size()/l.targets)
	if err != nil {
		return err
	}

	l.coefficients = coefficients

	return nil
}

// Apply applies the linearRegressor operator.
func (l *LinearRegressor) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	X := inputs[0]

	// The coefficients are float32, and so is the output for every input data type.
	if X.DataType() != tensor.Float32 {
		converted, err := ops.ConvertTensorDtype(X, int32(tensor.Float32))
		if err != nil {
			return nil, err
		}

		X = converted
	}

	Y, err := ops.Gemm(X, l.coefficients, l.intercepts, 1, 1, false, true)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{Y}, nil
}
