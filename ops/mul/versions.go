package mul

import "github.com/advancedclimatesystems/gonnx/ops"

var mulVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newMul, 7, mulTypeConstraints),
	13: ops.NewOperatorConstructor(newMul, 13, mulTypeConstraints),
}

func GetMulVersions() ops.OperatorVersions {
	return mulVersions
}
