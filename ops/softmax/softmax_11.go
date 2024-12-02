package softmax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Softmax11 represents the ONNX softmax operator.
type Softmax11 struct {
	// The axis along which to perform the Softmax11 operation.
	axis int
}

// newSoftmax11 creates a new softmax operator.
func newSoftmax11() ops.Operator {
	return &Softmax11{
		axis: -1, // ONNX definition says default 1, but our implementation like this yields the same results with this as default
	}
}

// Init initializes the softmax operator.
func (s *Softmax11) Init(n *onnx.NodeProto) error {
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
func (s *Softmax11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Softmax11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Softmax11) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Softmax11) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Softmax11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Softmax11) String() string {
	return "softmax11 operator"
}
