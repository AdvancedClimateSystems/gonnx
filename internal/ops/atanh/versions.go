package atanh

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
)

var atanhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newAtanh, 9, atanhTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return atanhVersions
}
