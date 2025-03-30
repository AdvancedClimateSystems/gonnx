package trilu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var triluTypeConstraints = [][]tensor.Dtype{ops.AllTypes, []tensor.Dtype{tensor.Int64}}

// Trilu represents the ONNX trilu operator.
type Trilu struct {
	ops.BaseOperator

	upper bool
}

// newTrilu creates a new trilu operator.
func newTrilu(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Trilu{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			2,
			typeConstraint,
			"trilu",
		),
		upper: true, // Default is true as per ONNX spec
	}
}

// Init initializes the trilu operator.
func (t *Trilu) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "upper":
			t.upper = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), t)
		}
	}

	return nil
}

// Apply applies the trilu operator.
func (t *Trilu) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]

	var k int

	if inputs[1] != nil {
		var err error

		k, err = ops.AnyToInt(inputs[1].ScalarValue())
		if err != nil {
			return nil, err
		}
	}

	rank := len(data.Shape())
	if rank < 2 {
		return nil, ops.ErrInvalidInput("input tensor must be at least rank 2", t.BaseOperator)
	}

	// Create output tensor with same shape and type as input
	out, ok := data.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", out.Clone())
	}

	zeroVal, err := ops.GetValueAsTensorType(0.0, out.Dtype())
	if err != nil {
		return nil, err
	}

	it := out.Iterator()
	it.Reset()

	for !it.Done() {
		coords := it.Coord()

		row := coords[rank-2]
		col := coords[rank-1]

		shouldZero := false
		if t.upper {
			shouldZero = col-row < k
		} else {
			shouldZero = col-row > k
		}

		if shouldZero {
			err = out.SetAt(zeroVal, coords...)
			if err != nil {
				return nil, err
			}
		}

		_, err := it.Next()
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, nil
}
