package or

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var orTypeConstraints = [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}

// Or represents the ONNX or operator.
type Or struct {
	ops.BaseOperator
}

// newOr creates a new or operator.
func newOr(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (o *Or) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Or,
		ops.MultidirectionalBroadcasting,
	)
}
