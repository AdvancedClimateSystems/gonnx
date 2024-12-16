package scaler

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var scalerTypeConstraints = [][]tensor.Dtype{
	{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

const (
	ScalerExpectedAttributes = 2
)

// Scaler represents the ONNX-ml scaler operator.
type Scaler struct {
	ops.BaseOperator

	offset tensor.Tensor
	scale  tensor.Tensor
}

// newScaler creates a new scaler operator.
func newScaler(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Scaler{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"scaler",
		),
	}
}

// Init initializes the scaler operator.
func (s *Scaler) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
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
