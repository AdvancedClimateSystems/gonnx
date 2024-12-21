package pow

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var pow7TypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var powTypeConstraints = [][]tensor.Dtype{
	{tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// Pow represents the ONNX pow operator.
type Pow struct {
	ops.BaseOperator
}

// newPow creates a new pow operator.
func newPow(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Pow{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"pow",
		),
	}
}

// Init initializes the pow operator.
func (a *Pow) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the pow operator.
func (a *Pow) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	powTensor := inputs[1]
	if inputs[0].Dtype() != powTensor.Dtype() {
		to, err := ops.DTypeToONNXType(inputs[0].Dtype())
		if err != nil {
			return nil, err
		}

		powTensor, err = ops.ConvertTensorDtype(powTensor, to)
		if err != nil {
			return nil, err
		}
	}

	return ops.ApplyBinaryOperation(
		inputs[0],
		powTensor,
		ops.Pow,
		ops.MultidirectionalBroadcasting,
	)
}
