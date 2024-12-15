package linearregressor

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var linearRegressorTypeConstraints = [][]tensor.Dtype{
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

	coefficients  tensor.Tensor
	intercepts    tensor.Tensor
	postTransform postTransformOption
	targets       int
}

// newLinearRegressor creates a new linearRegressor operator.
func newLinearRegressor(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
			l.coefficients = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
		case "intercepts":
			floats := attr.GetFloats()
			l.intercepts = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
		case "post_transform":
			return ops.ErrUnsupportedAttribute(attr.GetName(), l)
		case "targets":
			l.targets = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	err := l.coefficients.Reshape(l.targets, ops.NElements(l.coefficients.Shape()...)/l.targets)
	if err != nil {
		return err
	}

	return l.coefficients.T()
}

// Apply applies the linearRegressor operator.
func (l *LinearRegressor) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	X := inputs[0]

	result, err := tensor.MatMul(X, l.coefficients)
	if err != nil {
		return nil, err
	}

	result, intercepts, err := ops.UnidirectionalBroadcast(result, l.intercepts)
	if err != nil {
		return nil, err
	}

	Y, err := tensor.Add(result, intercepts)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y}, nil
}
