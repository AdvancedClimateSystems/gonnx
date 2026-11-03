package gemm

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var gemmTypeConstraints = [][]tensor.DataType{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

const (
	MinGemmInputs = 2
	MaxGemmInputs = 3
)

// Gemm represents the ONNX gemm operator.
type Gemm struct {
	ops.BaseOperator

	alpha  float32
	beta   float32
	transA bool
	transB bool
}

// newGemm creates a new gemm operator and initializes it with the default values.
func newGemm(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Gemm{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinGemmInputs,
			MaxGemmInputs,
			typeConstraints,
			"gemm",
		),
		alpha:  1.0,
		beta:   1.0,
		transA: false,
		transB: false,
	}
}

// Init initializes the Gemm operator based on the ModelProto attributes.
func (g *Gemm) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case alpha:
			g.alpha = attr.GetF()
		case beta:
			g.beta = attr.GetF()
		case transA:
			g.transA = ops.Int64ToBool(attr.GetI())
		case transB:
			g.transB = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the gemm operator on the given graph.
func (g *Gemm) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.Gemm(inputs[0], inputs[1], inputs[2], g.alpha, g.beta, g.transA, g.transB)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
