package gonnx

import (
	"errors"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/opset13"
)

var ErrInvalidOperator = errors.New("invalid operator getter")

// OpGetter is a function that gets an operator based on a string.
type OpGetter func(string) (ops.Operator, error)

var operatorGetters = map[int64]OpGetter{
	13: opset13.GetOperator,
}

// ResolveOperatorGetter resolves the getter for operators based on the opset version.
func ResolveOperatorGetter(opsetID int64) (OpGetter, error) {
	if getOperator, ok := operatorGetters[opsetID]; ok {
		return getOperator, nil
	}

	return nil, ErrInvalidOperator
}
