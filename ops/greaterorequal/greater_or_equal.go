package greaterorequal

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var greaterOrEqualTypeConstraints = [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}

// GreaterOrEqual represents the ONNX greaterOrEqual operator.
type GreaterOrEqual struct {
	ops.BaseOperator
}

// newGreaterOrEqual creates a new greaterOrEqual operator.
func newGreaterOrEqual(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &GreaterOrEqual{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"greaterorequal",
		),
	}
}

// Init initializes the greaterOrEqual operator.
func (g *GreaterOrEqual) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greaterOrEqual operator.
func (g *GreaterOrEqual) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gte,
		ops.MultidirectionalBroadcasting,
	)
}
