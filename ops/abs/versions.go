package abs

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var absVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newAbs(6, absTypeConstraint)),  // Same, but bfloat16 type is added
	13: ops.NewOperatorConstructor(newAbs(13, absTypeConstraint)), // Same, but bfloat16 type is added
}

func GetAbsVersions() ops.OperatorVersions {
	return absVersions
}
