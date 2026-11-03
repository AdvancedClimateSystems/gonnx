package mul

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var mulTypeConstraints = [][]tensor.DataType{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Mul represents the ONNX mul operator.
type Mul struct {
	ops.BaseOperator
}

// newMul creates a new mul operator.
func newMul(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Mul{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"mul",
		),
	}
}

// Init initializes the mul operator.
func (m *Mul) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the mul operator.
func (m *Mul) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Mul,
		ops.MultidirectionalBroadcasting,
	)
}
