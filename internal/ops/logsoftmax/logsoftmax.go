package logsoftmax

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var logSoftmaxTypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}}

// LogSoftmax represents the ONNX logsoftmax operator.
type LogSoftmax struct {
	ops.BaseOperator

	// The axis along which to perform the LogSoftmax operation.
	axis int
}

// newLogSoftmax creates a new logsoftmax operator.
func newLogSoftmax(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &LogSoftmax{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"logsoftmax",
		),
		axis: -1,
	}
}

// Init initializes the logsoftmax operator.
func (l *LogSoftmax) Init(n *onnx.NodeProto) error {
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
func (l *LogSoftmax) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	input := inputs[0]
	nDims := len(input.Shape())

	if l.axis < -nDims || l.axis >= nDims {
		return nil, ops.ErrAxisOutOfRange(-nDims, nDims, l.axis)
	}

	axis := l.axis
	if l.axis < 0 {
		axis += nDims
	}

	out, err := ops.Softmax(inputs[0], axis, true)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
