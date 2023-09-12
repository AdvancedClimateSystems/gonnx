package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// LogSoftmax represents the ONNX logsoftmax operator.
type LogSoftmax struct {
	// The axis along which to perform the LogSoftmax operation.
	axis int
}

// newLogSoftmax creates a new logsoftmax operator.
func newLogSoftmax() ops.Operator {
	return &LogSoftmax{
		axis: -1,
	}
}

// Init initializes the logsoftmax operator.
func (l *LogSoftmax) Init(attributes []*onnx.AttributeProto) error {
	nAttributes := len(attributes)
	if nAttributes > 1 {
		return fmt.Errorf(ops.InvalidAttrCountErrTemplate, l, 1, nAttributes)
	}

	if nAttributes == 1 {
		l.axis = int(attributes[0].GetI())
	}

	return nil
}

// Apply applies the logsoftmax operator.
func (l *LogSoftmax) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]
	nDims := len(input.Shape())

	if l.axis < -nDims || l.axis >= nDims {
		return nil, fmt.Errorf(ops.AxisOutOfRangeErrTemplate, nDims, nDims, l.axis)
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
func (l *LogSoftmax) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(l, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (l *LogSoftmax) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (l *LogSoftmax) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (l *LogSoftmax) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (l *LogSoftmax) String() string {
	return "logsoftmax operator"
}
