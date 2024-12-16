package gemm

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gemmTypeConstraints = [][]tensor.Dtype{
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
func newGemm(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (g *Gemm) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

	// If C was not given, it is assumed to be 0, hence we can stop the calculation here.
	if c == nil {
		return []tensor.Tensor{x}, nil
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
