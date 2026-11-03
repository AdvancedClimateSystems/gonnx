package or

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var orTypeConstraints = [][]tensor.DataType{{tensor.Bool}, {tensor.Bool}}

// Or represents the ONNX or operator.
type Or struct {
	ops.BaseOperator
}

// newOr creates a new or operator.
func newOr(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Or{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"or",
		),
	}
}

// Init initializes the or operator.
func (o *Or) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the or operator.
func (o *Or) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Or,
		ops.MultidirectionalBroadcasting,
	)
}
