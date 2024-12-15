package div

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var divTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Div represents the ONNX div operator.
type Div struct {
	ops.BaseOperator
}

// newDiv creates a new div operator.
func newDiv(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Div{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"div",
		),
	}
}

// Init initializes the div operator.
func (d *Div) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the div operator.
func (d *Div) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ApplyBinaryOperation(
		inputs[0],
		inputs[1],
		ops.Div,
		ops.MultidirectionalBroadcasting,
	)
}
