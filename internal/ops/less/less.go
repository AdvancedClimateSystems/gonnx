package less

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var less7TypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}

var lessTypeConstraints = [][]tensor.DataType{ops.AllTypes, ops.AllTypes}

// Less represents the ONNX less operator.
type Less struct {
	ops.BaseOperator
}

// newLess creates a new less operator.
func newLess(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Less{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"less",
		),
	}
}

// Init initializes the less operator.
func (l *Less) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the less operator.
func (l *Less) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}
