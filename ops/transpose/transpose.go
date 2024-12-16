package transpose

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var transposeTypeConstraint = [][]tensor.Dtype{ops.AllTypes}

// Transpose represents the ONNX transpose operator.
type Transpose struct {
	ops.BaseOperator

	perm []int
}

// newTranspose creates a new transpose operator.
func newTranspose(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Transpose{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"transpose",
		),
	}
}

// Init initializes the transpose operator.
func (t *Transpose) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) == 1 {
		attr := attributes[0]

		if attr.GetName() != "perm" {
			return ops.ErrInvalidAttribute(attr.GetName(), t)
		}

		attrPerm := attr.GetInts()

		perm := make([]int, 0)
		for _, val := range attrPerm {
			perm = append(perm, int(val))
		}

		t.perm = perm
	}

	return nil
}

// Apply applies the transpose operator.
func (t *Transpose) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Transpose(inputs[0], t.perm...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
