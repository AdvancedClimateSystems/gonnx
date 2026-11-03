package matmul

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var matmul1TypeConstraints = [][]tensor.DataType{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var matmulTypeConstraints = [][]tensor.DataType{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// MatMul represents the ONNX matmul operator.
type MatMul struct {
	ops.BaseOperator

	weights *ops.PreparedWeights
}

// newMatMul returns a new MatMul operator.
func newMatMul(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &MatMul{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"matmul",
		),
	}
}

// Init initializes the matmul operator.
func (m *MatMul) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the matmul operator.
func (m *MatMul) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if packed := m.weights.For(inputs[1]); packed != nil && inputs[0].DataType() == tensor.Float32 {
		out, err := ops.MatMulPacked(inputs[0], packed)
		if err != nil {
			return nil, err
		}

		return []*tensor.Tensor{out}, nil
	}

	out, err := ops.MatMul(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}

// Prepare packs the second input when it is a 2D parameter of the model.
func (m *MatMul) Prepare(constants []*tensor.Tensor) {
	m.weights = ops.PrepareWeights(constants[1], false)
}
