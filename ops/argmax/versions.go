package argmax

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var argMaxVersions = ops.OperatorVersions{
	11: ops.NewOperatorConstructor(newArgMax, 11, argMaxTypeConstraints),
	12: ops.NewOperatorConstructor(newArgMax, 12, argMaxTypeConstraints),
	13: ops.NewOperatorConstructor(newArgMax, 13, argMaxTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return argMaxVersions
}
