package xor

import "github.com/advancedclimatesystems/gonnx/ops"

var xorVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newXor(7, xorTypeConstraint)),
}

func GetXorVersions() ops.OperatorVersions {
	return xorVersions
}
