package sqrt

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var sqrtTypeConstraints = [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}

// Sqrt represents the ONNX sqrt operator.
type Sqrt struct {
	ops.BaseOperator
}

// newSqrt creates a new sqrt operator.
func newSqrt(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Sqrt{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"sqrt",
		),
	}
}

// Init initializes the sqrt operator.
func (s *Sqrt) Init(_ *onnx.NodeProto) error {
	return nil
}

// Apply applies the sqrt operator.
func (s *Sqrt) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, err := tensor.Sqrt(inputs[0])
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}
