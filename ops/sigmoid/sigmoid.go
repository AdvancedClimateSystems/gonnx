package sigmoid

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var sigmoidTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Sigmoid represents the ONNX sigmoid operator.
type Sigmoid struct {
	ops.BaseOperator
}

// newSigmoid returns a new sigmoid operator.
func newSigmoid(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Sigmoid{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"sigmoid",
		),
	}
}

// Init initializes the sigmoid operator.
func (s *Sigmoid) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (s *Sigmoid) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Sigmoid(inputs[0])

	return []tensor.Tensor{out}, err
}
