package identity

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var identityTypeConstraints = [][]tensor.DataType{ops.AllTypes}

// Identity represents the ONNX identity operator.
type Identity struct {
	ops.BaseOperator
}

// newIdentity creates a new identity operator.
func newIdentity(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (a *Identity) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{inputs[0]}, nil
}
