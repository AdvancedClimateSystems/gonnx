package scaler

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	Scaler1ExpectedAttributes = 2
	MinScaler1Inputs          = 1
	MaxScaler1Inputs          = 1
)

// Scaler1 represents the ONNX-ml scaler operator.
type Scaler1 struct {
	offset tensor.Tensor
	scale  tensor.Tensor
}

// newScaler1 creates a new scaler operator.
func newScaler1() ops.Operator {
	return &Scaler1{}
}

// Init initializes the scaler operator.
func (s *Scaler1) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	if len(attributes) != Scaler1ExpectedAttributes {
		return ops.ErrInvalidAttributeCount(Scaler1ExpectedAttributes, len(attributes), s)
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
func (s *Scaler1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (s *Scaler1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Scaler1) GetMinInputs() int {
	return MinScaler1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Scaler1) GetMaxInputs() int {
	return MaxScaler1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Scaler1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Scaler1) String() string {
	return "scaler1 operator"
}
