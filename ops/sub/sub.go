package sub

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var subTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Sub represents the ONNX sub operator.
type Sub struct {
	ops.BaseOperator
}

// newSub creates a new sub operator.
func newSub(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Sub{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"sub",
		),
	}
}

// Init initializes the sub operator.
func (s *Sub) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the sub operator.
func (s *Sub) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Sub,
		ops.MultidirectionalBroadcasting,
	)
}
