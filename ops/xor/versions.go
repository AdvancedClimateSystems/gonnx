package xor

import "github.com/advancedclimatesystems/gonnx/ops"

var XorVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newXor(7, xorTypeConstraint)),
}
