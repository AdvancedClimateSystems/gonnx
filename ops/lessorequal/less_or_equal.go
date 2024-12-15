package lessorequal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var lessOrEqualTypeConstraints = [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}

// LessOrEqual represents the ONNX lessOrEqual operator.
type LessOrEqual struct {
	ops.BaseOperator
}

// newLessOrEqual creates a new lessOrEqual operator.
func newLessOrEqual(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &LessOrEqual{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"lessorequal",
		),
	}
}

// Init initializes the lessOrEqual operator.
func (l *LessOrEqual) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the lessOrEqual operator.
func (l *LessOrEqual) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lte,
		ops.MultidirectionalBroadcasting,
	)
}
