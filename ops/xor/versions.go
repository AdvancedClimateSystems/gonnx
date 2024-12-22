package xor

import "github.com/advancedclimatesystems/gonnx/ops"

var xorVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newXor, 7, xorTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return xorVersions
}
