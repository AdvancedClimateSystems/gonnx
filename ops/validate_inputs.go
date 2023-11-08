package ops

import (
	"gorgonia.org/tensor"
)

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
	padLength := 0
	min := op.GetMinInputs()
	max := op.GetMaxInputs()

	if min == max {
		if nInputs != min {
			return 0, ErrInvalidInputCount(nInputs, op)
		}

		padLength = min
	} else {
		if nInputs < min || nInputs > max {
			return 0, ErrInvalidOptionalInputCount(nInputs, op)
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
			return ErrInvalidInputType(i, input.Dtype().Name(), op)
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
