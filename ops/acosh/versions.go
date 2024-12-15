package acosh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var acoshVersions = ops.OperatorVersions{
	9: newAcosh,
}

func GetAcoshVersions() ops.OperatorVersions {
	return acoshVersions
}
