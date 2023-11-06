package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	ScalerExpectedAttributes = 2
	ScalerInputs             = 1
)

// Scaler represents the ONNX-ml scaler operator.
type Scaler struct {
	offset tensor.Tensor
	scale  tensor.Tensor
}

// newScaler creates a new scaler operator.
func newScaler() ops.Operator {
	return &Scaler{}
}

// Init initializes the scaler operator.
func (s *Scaler) Init(attributes []*onnx.AttributeProto) error {
	if len(attributes) != ScalerExpectedAttributes {
		return ops.ErrInvalidAttributeCount(ScalerExpectedAttributes, len(attributes), s)
	}

	for _, attr := range attributes {
		switch attr.GetName() {
		case "offset":
			floats := attr.GetFloats()
			s.offset = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
		case "scale":
			floats := attr.GetFloats()
			s.scale = tensor.New(tensor.WithShape(len(floats)), tensor.WithBacking(floats))
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), s)
		}
	}

	return nil
}

// Apply applies the scaler operator.
func (s *Scaler) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	X, offset, err := ops.UnidirectionalBroadcast(inputs[0], s.offset)
	if err != nil {
		return nil, err
	}

	X, err = tensor.Sub(X, offset)
	if err != nil {
		return nil, err
	}

	X, scale, err := ops.UnidirectionalBroadcast(X, s.scale)
	if err != nil {
		return nil, err
	}

	Y, err := tensor.Mul(X, scale)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Scaler) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Scaler) GetMinInputs() int {
	return ScalerInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Scaler) GetMaxInputs() int {
	return ScalerInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Scaler) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Scaler) String() string {
	return "scaler operator"
}
