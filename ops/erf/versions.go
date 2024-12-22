package erf

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var erfVersions = ops.OperatorVersions{
	9:  ops.NewOperatorConstructor(newErf, 9, erfTypeConstraints),
	13: ops.NewOperatorConstructor(newErf, 13, erfTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return erfVersions
}
