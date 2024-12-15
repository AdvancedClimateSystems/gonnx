package not

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var notTypeConstraints = [][]tensor.Dtype{{tensor.Bool}}

// Not represents the ONNX not operator.
type Not struct {
	ops.BaseOperator
}

// newNot creates a new not operator.
func newNot(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Not{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"not",
		),
	}
}

// Init initializes the not operator.
func (n *Not) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the not operator.
func (n *Not) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := inputs[0].Apply(not)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

func not(x bool) bool {
	return !x
}
