package linearregressor

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinLinearRegressor1Inputs = 1
	MaxLinearRegressor1Inputs = 1
)

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

// LinearRegressor1 represents the ONNX-ml linearRegressor operator.
type LinearRegressor1 struct {
	coefficients  tensor.Tensor
	intercepts    tensor.Tensor
	postTransform postTransformOption
	targets       int
}

// newLinearRegressor1 creates a new linearRegressor operator.
func newLinearRegressor1() ops.Operator {
	return &LinearRegressor1{
		postTransform: noTransform,
		targets:       1,
	}
}

// Init initializes the linearRegressor operator.
func (l *LinearRegressor1) Init(n *onnx.NodeProto) error {
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
func (l *LinearRegressor1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LinearRegressor1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LinearRegressor1) GetMinInputs() int {
	return MinLinearRegressor1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LinearRegressor1) GetMaxInputs() int {
	return MaxLinearRegressor1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LinearRegressor1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LinearRegressor1) String() string {
	return "linearregressor1 operator"
}
