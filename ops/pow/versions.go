package pow

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var powVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newPow, 7, pow7TypeConstraints),
	12: ops.NewOperatorConstructor(newPow, 12, powTypeConstraints),
	13: ops.NewOperatorConstructor(newPow, 13, powTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return powVersions
}
