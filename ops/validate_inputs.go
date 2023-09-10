package ops

import (
	"fmt"

	"gorgonia.org/tensor"
)

// AllTypes is a type constraint which allows all types.
var AllTypes = []tensor.Dtype{
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Float32, tensor.Float64,
	tensor.Complex64, tensor.Complex128,
	tensor.String,
	tensor.Bool,
}

// ValidateInputs validates if a list of nodes has enough (not too few or too many) nodes.
// When there are fewer input nodes then the given max, the list is padded with nils.
// Expects either 1 requirement ==> the expected number of inputs, or 2 requirements,
// the minimum and the maximum number of inputs.
func ValidateInputs(op Operator, inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	padLength, err := checkNInputs(op, inputs)
	if err != nil {
		return inputs, err
	}

	inputs = padInputs(inputs, padLength)

	err = checkInputTypes(op, inputs)
	if err != nil {
		return inputs, err
	}

	return inputs, nil
}

func checkNInputs(op Operator, inputs []tensor.Tensor) (int, error) {
	nInputs := len(inputs)

	var padLength int

	min := op.GetMinInputs()
	max := op.GetMaxInputs()

	if min == max {
		if nInputs != min {
			return 0, fmt.Errorf(InvalidInputCountErrTemplate, op, min, nInputs)
		}
		padLength = min
	} else {
		if nInputs < min || nInputs > max {
			return 0, fmt.Errorf(InvalidOptionalInputCountErrTemplate, op, min, max, nInputs)
		}
		padLength = max
	}

	return padLength, nil
}

// padInputs pads a list of input nodes to the given length with nils.
func padInputs(inputs []tensor.Tensor, length int) []tensor.Tensor {
	for len(inputs) < length {
		inputs = append(inputs, nil)
	}
	return inputs
}

func checkInputTypes(op Operator, inputs []tensor.Tensor) error {
	typeConstraints := op.GetInputTypeConstraints()
	for i, input := range inputs {
		// Optional inputs can be nil, we can not check for type constraints then.
		if input == nil {
			continue
		}

		typeConstraint := newTypeConstraint(typeConstraints[i])

		if _, ok := typeConstraint[input.Dtype()]; !ok {
			return fmt.Errorf("%v: input %d does not allow type %v", op, i, input.Dtype())
		}
	}
	return nil
}

// newTypeConstraint creates a map with for every type whether or not they are allowed.
func newTypeConstraint(allowedTypes []tensor.Dtype) map[tensor.Dtype]bool {
	typeConstraint := make(map[tensor.Dtype]bool)

	for _, allowedType := range allowedTypes {
		typeConstraint[allowedType] = true
	}
	return typeConstraint
}
