package ops

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Concrete implementation for shared operator methods.
type BaseOperator struct {
	name                 string
	version              int
	minInputs            int
	maxInputs            int
	inputTypeConstraints [][]tensor.DataType
}

func NewBaseOperator(version, minInputs, maxInputs int, inputTypeConstraints [][]tensor.DataType, name string) BaseOperator {
	return BaseOperator{
		name:                 name,
		version:              version,
		minInputs:            minInputs,
		maxInputs:            maxInputs,
		inputTypeConstraints: inputTypeConstraints,
	}
}

// ValidateInputs validates the inputs for the operator.
func (f BaseOperator) ValidateInputs(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return ValidateInputs(f, inputs)
}

// Version returns the version of the operator.
func (f BaseOperator) Version() int {
	return f.version
}

// GetMinInputs returns the minimum number of input tensors.
func (f BaseOperator) GetMinInputs() int {
	return f.minInputs
}

// GetMaxInputs returns the maximum number of input tensors.
func (f BaseOperator) GetMaxInputs() int {
	return f.maxInputs
}

// GetInputTypeConstraints returns allowed input types.
func (f BaseOperator) GetInputTypeConstraints() [][]tensor.DataType {
	return f.inputTypeConstraints
}

func (f BaseOperator) String() string {
	return fmt.Sprintf("%s v%d", f.name, f.version)
}
