package not

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var notTypeConstraints = [][]tensor.DataType{{tensor.Bool}}

// Not represents the ONNX not operator.
type Not struct {
	ops.BaseOperator
}

// newNot creates a new not operator.
func newNot(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (n *Not) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{ops.Map(inputs[0], not)}, nil
}

func not(x bool) bool {
	return !x
}
