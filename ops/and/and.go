package and

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var andTypeConstraints = [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}

// And represents the ONNX and operator.
type And struct {
	ops.BaseOperator
}

// newAnd creates a new and operator.
func newAnd(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &And{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"and",
		),
	}
}

// Init initializes the and operator.
func (a *And) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the and operator.
func (a *And) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.And,
		ops.MultidirectionalBroadcasting,
	)
}
