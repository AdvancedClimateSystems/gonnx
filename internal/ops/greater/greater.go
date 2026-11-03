package greater

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var greater7TypeConstraints = [][]tensor.DataType{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}

var greaterTypeConstraints = [][]tensor.DataType{ops.AllTypes, ops.AllTypes}

// Greater represents the ONNX greater operator.
type Greater struct {
	ops.BaseOperator
}

// newGreater creates a new greater operator.
func newGreater(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Greater{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"greater",
		),
	}
}

// Init initializes the greater operator.
func (g *Greater) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the greater operator.
func (g *Greater) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Gt,
		ops.MultidirectionalBroadcasting,
	)
}
