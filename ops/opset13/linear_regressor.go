package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinLinearRegressorInputs = 1
	MaxLinearRegressorInputs = 1
)

type PostTransformOption string

const (
	NoTransform          PostTransformOption = "NONE"
	SoftmaxTransform     PostTransformOption = "SOFTMAX"
	LogisticTransform    PostTransformOption = "LOGISTIC"
	SoftmaxZeroTransform PostTransformOption = "SOFTMAX_ZERO"
	ProbitTransform      PostTransformOption = "PROBIT"
)

// LinearRegressor represents the ONNX-ml linearRegressor operator.
type LinearRegressor struct {
	coefficients  tensor.Tensor
	intercepts    tensor.Tensor
	postTransform PostTransformOption
	targets       int
}

// newLinearRegressor creates a new linearRegressor operator.
func newLinearRegressor() ops.Operator {
	return &LinearRegressor{
		postTransform: NoTransform,
		targets:       1,
	}
}

// Init initializes the linearRegressor operator.
func (l *LinearRegressor) Init(attributes []*onnx.AttributeProto) error {
	for _, attr := range attributes {
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LinearRegressor) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LinearRegressor) GetMinInputs() int {
	return MinLinearRegressorInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LinearRegressor) GetMaxInputs() int {
	return MaxLinearRegressorInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LinearRegressor) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LinearRegressor) String() string {
	return "linearRegressor operator"
}
