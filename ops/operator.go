package ops

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"gorgonia.org/tensor"
)

// Operator is the base interface for all operators.
type Operator interface {
	// String should return a simple string describing the operator
	String() string

	// Init should initialize the operator based on the given attributes. How these
	// attributes influence the operator is defined by the ONNX standard, and can be
	// found in https://github.com/onnx/onnx/blob/main/docs/Operators.md
	Init([]*onnx.AttributeProto) error

	// Apply should apply the operator to the list of input tensors. It should return a
	// list with output tensors, the result of the operator.
	Apply([]tensor.Tensor) ([]tensor.Tensor, error)

	// GetMinInputs should return the minimum number of inputs this operator expects.
	GetMinInputs() int

	// GetMaxInputs should return the maximum number of inputs this operator expects.
	GetMaxInputs() int

	// GetInputTypeConstraints should return a list. Every element represents a set of
	// allowed tensor dtypes for the corresponding input tensor.
	GetInputTypeConstraints() [][]tensor.Dtype

	// ValidateInputs should validate the list of input tensors. It should check for both
	// the right amount of inputs and the correct dtypes of the tensors.
	ValidateInputs([]tensor.Tensor) ([]tensor.Tensor, error)
}
