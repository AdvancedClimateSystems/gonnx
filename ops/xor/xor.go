package xor

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var xorTypeConstraints = [][]tensor.Dtype{{tensor.Bool}, {tensor.Bool}}

// Xor represents the ONNX xor operator.
type Xor struct {
	ops.BaseOperator
}

// newXor creates a new xor operator.
func newXor(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Xor{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraint,
			"xor",
		),
	}
}

// Init initializes the xor operator.
func (x *Xor) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the xor operator.
func (x *Xor) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Xor,
		ops.MultidirectionalBroadcasting,
	)
}
