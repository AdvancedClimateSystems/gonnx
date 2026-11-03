package scaler

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var scalerTypeConstraints = [][]tensor.DataType{
	{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

const (
	ScalerExpectedAttributes = 2
)

// Scaler represents the ONNX-ml scaler operator.
type Scaler struct {
	ops.BaseOperator

	offset *tensor.Tensor
	scale  *tensor.Tensor
}

// newScaler creates a new scaler operator.
func newScaler(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
			s.offset = tensor.New(floats, len(floats))
		case "scale":
			floats := attr.GetFloats()
			s.scale = tensor.New(floats, len(floats))
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), s)
		}
	}

	return nil
}

// Apply applies the scaler operator.
func (s *Scaler) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	X, err := toFloat32(inputs[0])
	if err != nil {
		return nil, err
	}

	// The offset and scale are vectors, so broadcasting them to X repeats them along the
	// last dimension of X.
	for _, t := range []*tensor.Tensor{s.offset, s.scale} {
		if _, err := ops.UnidirectionalBroadcastShape(X.Shape(), t.Shape()); err != nil {
			return nil, err
		}
	}

	x := tensor.Data[float32](X)
	offsets, scales := tensor.Data[float32](s.offset), tensor.Data[float32](s.scale)
	out := make([]float32, len(x))
	offsetIndex, scaleIndex := 0, 0

	for i, value := range x {
		out[i] = (value - offsets[offsetIndex]) * scales[scaleIndex]

		if offsetIndex++; offsetIndex == len(offsets) {
			offsetIndex = 0
		}

		if scaleIndex++; scaleIndex == len(scales) {
			scaleIndex = 0
		}
	}

	return []*tensor.Tensor{tensor.New(out, X.Shape()...)}, nil
}

// toFloat32 converts a tensor to float32, as the Scaler output is float32 for every input
// data type.
func toFloat32(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.DataType() == tensor.Float32 {
		return t, nil
	}

	return ops.ConvertTensorDtype(t, int32(tensor.Float32))
}
