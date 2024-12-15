package less

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var less7TypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}

var lessTypeConstraints = [][]tensor.Dtype{ops.AllTypes, ops.AllTypes}

// Less represents the ONNX less operator.
type Less struct {
	ops.BaseOperator
}

// newLess creates a new less operator.
func newLess(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (l *Less) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Lt,
		ops.MultidirectionalBroadcasting,
	)
}
