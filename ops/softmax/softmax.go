package softmax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var softmaxTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Softmax represents the ONNX softmax operator.
type Softmax struct {
	ops.BaseOperator

	// The axis along which to perform the Softmax operation.
	axis int
}

// newSoftmax creates a new softmax operator.
func newSoftmax(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Softmax{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"softmax",
		),
		axis: -1,
	}
}

// Init initializes the softmax operator.
func (s *Softmax) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	nAttributes := len(attributes)

	if nAttributes > 1 {
		return ops.ErrInvalidAttributeCount(1, nAttributes, s)
	}

	if nAttributes == 1 {
		s.axis = int(attributes[0].GetI())
	}

	return nil
}

// Apply applies the softmax operator.
func (s *Softmax) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]
	nDims := len(input.Shape())

	if s.axis < -nDims || s.axis >= nDims {
		return nil, ops.ErrAxisOutOfRange(-nDims, nDims, s.axis)
	}

	axis := s.axis
	if s.axis < 0 {
		axis += nDims
	}

	out, err := tensor.SoftMax(inputs[0], axis)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
