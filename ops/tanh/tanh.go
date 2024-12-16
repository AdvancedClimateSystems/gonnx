package tanh

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var tanhTypeConstraint = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
}

// Tanh represents the tanh operator.
type Tanh struct {
	ops.BaseOperator
}

// newTanh returns a new tanh operator.
func newTanh(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Tanh{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraint,
			"tanh",
		),
	}
}

// Init initializes the sigmoid operator.
func (t *Tanh) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the sigmoid operator to the input node.
func (t *Tanh) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.Tanh(inputs[0])

	return []tensor.Tensor{out}, err
}
