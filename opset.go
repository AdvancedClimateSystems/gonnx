package gonnx

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/opset13"
)

// OpGetter is a function that gets an operator based on a string.
type OpGetter func(string) (ops.Operator, error)

var operatorGetters = map[int64]OpGetter{
	13: opset13.GetOperator,
}

// ResolveOperatorGetter resolves the getter for operators based on the opset version.
func ResolveOperatorGetter(opsetID int64) (OpGetter, error) {
	if GetOperator, ok := operatorGetters[opsetID]; ok {
		return GetOperator, nil
	}

	opsets := make([]int64, len(operatorGetters))
	for version := range operatorGetters {
		opsets = append(opsets, version)
	}

	return nil, fmt.Errorf("expected opset to be in %d, got operator set %d", opsets, opsetID)
}
