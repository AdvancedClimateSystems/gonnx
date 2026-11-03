package gemm

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Gemm9 represents the ONNX gemm operator, for version <= 9.
type Gemm9 struct {
	ops.BaseOperator

	alpha  float32
	beta   float32
	transA bool
	transB bool

	weights *ops.PreparedWeights
}

// newGemm7 creates a new gemm operator and initializes it with the default values.
func newGemm9(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Gemm9{
		BaseOperator: ops.NewBaseOperator(version, 3, 3, typeConstraints, "gemm"),
		alpha:        1.0,
		beta:         1.0,
		transA:       false,
		transB:       false,
	}
}

// Init initializes the Gemm7 operator based on the ModelProto attributes.
func (g *Gemm9) Init(n *onnx.NodeProto) error {
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
func (g *Gemm9) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if packed := g.weights.For(inputs[1]); packed != nil && !g.transA && inputs[0].DataType() == tensor.Float32 {
		out, err := ops.GemmPacked(inputs[0], packed, inputs[2], g.alpha, g.beta)
		if err != nil {
			return nil, err
		}

		return []*tensor.Tensor{out}, nil
	}

	out, err := ops.Gemm(inputs[0], inputs[1], inputs[2], g.alpha, g.beta, g.transA, g.transB)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}

// Prepare packs the weights B when they are a parameter of the model.
func (g *Gemm9) Prepare(constants []*tensor.Tensor) {
	g.weights = ops.PrepareWeights(constants[1], g.transB)
}
