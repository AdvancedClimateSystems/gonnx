package relu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reluTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Relu represents the ONNX relu operator.
type Relu struct {
	ops.BaseOperator
}

// newRelu creates a new relu operator.
func newRelu(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Relu{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"relu",
		),
	}
}

// Init initializes the relu operator.
func (r *Relu) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the relu operator.
func (r *Relu) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := ops.ReLU(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
