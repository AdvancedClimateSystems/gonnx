package logsoftmax

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// LogSoftmax13 represents the ONNX logsoftmax operator.
type LogSoftmax13 struct {
	// The axis along which to perform the LogSoftmax13 operation.
	axis int
}

// newLogSoftmax13 creates a new logsoftmax operator.
func newLogSoftmax13() ops.Operator {
	return &LogSoftmax13{
		axis: -1,
	}
}

// Init initializes the logsoftmax operator.
func (l *LogSoftmax13) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	nAttributes := len(attributes)
	if nAttributes > 1 {
		return ops.ErrInvalidAttributeCount(1, nAttributes, l)
	}

	if nAttributes == 1 {
		l.axis = int(attributes[0].GetI())
	}

	return nil
}

// Apply applies the logsoftmax operator.
func (l *LogSoftmax13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]
	nDims := len(input.Shape())

	if l.axis < -nDims || l.axis >= nDims {
		return nil, ops.ErrAxisOutOfRange(-nDims, nDims, l.axis)
	}

	axis := l.axis
	if l.axis < 0 {
		axis += nDims
	}

	out, err := tensor.LogSoftMax(inputs[0], axis)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (l *LogSoftmax13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LogSoftmax13) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LogSoftmax13) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LogSoftmax13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LogSoftmax13) String() string {
	return "logsoftmax13 operator"
}
