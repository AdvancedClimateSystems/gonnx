package abs

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var absTypeConstraint = [][]tensor.Dtype{
	{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Abs represents the ONNX abs operator.
type Abs struct {
	ops.BaseOperator
}

// newAbs creates a new abs operator.
func newAbs(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Abs{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"abs",
		),
	}
}

// Init initializes the abs operator.
func (a *Abs) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the abs operator.
func (a *Abs) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Abs(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
