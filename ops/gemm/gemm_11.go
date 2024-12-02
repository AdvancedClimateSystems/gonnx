package gemm

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinGemm11Inputs = 2
	MaxGemm11Inputs = 3
)

// Gemm11 represents the ONNX gemm operator.
type Gemm11 struct {
	alpha  float32
	beta   float32
	transA bool
	transB bool
}

// newGemm11 creates a new gemm operator and initializes it with the default values.
func newGemm11() ops.Operator {
	return &Gemm11{
		alpha:  1.0,
		beta:   1.0,
		transA: false,
		transB: false,
	}
}

// Init initializes the Gemm11 operator based on the ModelProto attributes.
func (g *Gemm11) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "alpha":
			g.alpha = attr.GetF()
		case "beta":
			g.beta = attr.GetF()
		case "transA":
			g.transA = ops.Int64ToBool(attr.GetI())
		case "transB":
			g.transB = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the gemm operator on the given graph.
func (g *Gemm11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Gemm11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Gemm11) GetMinInputs() int {
	return MinGemm11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Gemm11) GetMaxInputs() int {
	return MaxGemm11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Gemm11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
		{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Gemm11) String() string {
	return "gemm11 operator"
}
