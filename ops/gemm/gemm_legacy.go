package gemm

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Gemm9 represents the ONNX gemm operator, for version <= 9.
type Gemm9 struct {
	ops.BaseOperator

	alpha  float32
	beta   float32
	transA bool
	transB bool
}

// newGemm7 creates a new gemm operator and initializes it with the default values.
func newGemm9(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (g *Gemm9) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var err error

	a := inputs[0]
	b := inputs[1]
	c := inputs[2]

	if g.transA {
		a, err = tensor.Transpose(a)
		if err != nil {
			return nil, err
		}
	}

	if g.transB {
		b, err = tensor.Transpose(b)
		if err != nil {
			return nil, err
		}
	}

	x, err := tensor.MatMul(a, b)
	if err != nil {
		return nil, err
	}

	x, err = tensor.Mul(x, g.alpha)
	if err != nil {
		return nil, err
	}

	y, err := tensor.Mul(c, g.beta)
	if err != nil {
		return nil, err
	}

	x, y, err = ops.UnidirectionalBroadcast(x, y)
	if err != nil {
		return nil, err
	}

	output, err := tensor.Add(x, y)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{output}, nil
}
