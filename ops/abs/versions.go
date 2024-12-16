package abs

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var absVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newAbs, 6, absTypeConstraint),
	13: ops.NewOperatorConstructor(newAbs, 13, absTypeConstraint),
}

func GetAbsVersions() ops.OperatorVersions {
	return absVersions
}
