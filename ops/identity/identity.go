package identity

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var identityTypeConstraints = [][]tensor.Dtype{ops.AllTypes}

// Identity represents the ONNX identity operator.
type Identity struct {
	ops.BaseOperator
}

// newIdentity creates a new identity operator.
func newIdentity(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Identity{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"identity",
		),
	}
}

// Init initializes the identity operator.
func (a *Identity) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the identity operator.
func (a *Identity) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	return []tensor.Tensor{out}, nil
}
