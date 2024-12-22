package acos

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var acosVersions = ops.OperatorVersions{
	7: newAcos,
}

func GetVersions() ops.OperatorVersions {
	return acosVersions
}
