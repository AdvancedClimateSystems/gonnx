package add

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var addTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Add represents the ONNX add operator.
type Add struct {
	ops.BaseOperator
}

// newAdd creates a new add operator.
func newAdd(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Add{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"add",
		),
	}
}

// Init initializes the add operator.
func (a *Add) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the add operator.
func (a *Add) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Add,
		ops.MultidirectionalBroadcasting,
	)
}
