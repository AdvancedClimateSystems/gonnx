package atanh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var atanhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newAtanh, 9, atanhTypeConstraints),
}

func GetAtanhVersions() ops.OperatorVersions {
	return atanhVersions
}
