package mul

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var mulVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newMul, 7, mulTypeConstraints),
	13: ops.NewOperatorConstructor(newMul, 13, mulTypeConstraints),
	14: ops.NewOperatorConstructor(newMul, 14, mul14TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return mulVersions
}

var mul14TypeConstraints = [][]tensor.Dtype{
	ops.NumericTypes,
	ops.NumericTypes,
}
