package equal

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var equal7TypeConstraints = [][]tensor.DataType{
	{tensor.Bool, tensor.Int32, tensor.Int64},
	{tensor.Bool, tensor.Int32, tensor.Int64},
}

var equalTypeConstraints = [][]tensor.DataType{ops.AllTypes, ops.AllTypes}

// Equal represents the ONNX equal operator.
type Equal struct {
	ops.BaseOperator
}

// newEqual creates a new equal operator.
func newEqual(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Equal{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"equal",
		),
	}
}

// Init initializes the equal operator.
func (e *Equal) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the equal operator.
func (e *Equal) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Equal,
		ops.MultidirectionalBroadcasting,
	)
}
