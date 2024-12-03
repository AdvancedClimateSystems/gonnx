package xor

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var xorTypeConstraint = [][]tensor.Dtype{
	{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Xor represents the ONNX xor operator.
type Xor struct {
	ops.BaseOperator
}

// newXor creates a new xor operator.
func newXor(version int, typeConstraint [][]tensor.Dtype) *Xor {
	return &Xor{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"xor",
		),
	}
}

// Init initializes the xor operator.
func (a *Xor) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the xor operator.
func (a *Xor) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Xor,
		ops.MultidirectionalBroadcasting,
	)
}
