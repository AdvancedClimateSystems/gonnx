package cumsum

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var cumsumTypeConstraints = [][]tensor.DataType{
	{tensor.Int32, tensor.Int64, tensor.Uint32, tensor.Uint64, tensor.Float32, tensor.Float64},
	{tensor.Int32, tensor.Int64},
}

// CumSum represents the ONNX cumsum operator.
type CumSum struct {
	ops.BaseOperator

	exclusive bool
	reverse   bool
}

// newCumSum creates a new cumsum operator.
func newCumSum(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &CumSum{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"cumsum",
		),
		exclusive: false,
		reverse:   false,
	}
}

// Init initializes the cumsum operator.
func (c *CumSum) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "exclusive":
			c.exclusive = attr.GetI() == 1
		case "reverse":
			c.reverse = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), c)
		}
	}

	return nil
}

// Apply applies the cumsum operator.
func (c *CumSum) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	axes, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	if len(axes) != 1 {
		return nil, ops.ErrInvalidInput("axis must be a scalar or a tensor with one element", c.BaseOperator)
	}

	out, err := ops.CumSum(inputs[0], axes[0], c.exclusive, c.reverse)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}
