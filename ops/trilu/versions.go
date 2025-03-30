package trilu

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var triluVersions = ops.OperatorVersions{
	14: ops.NewOperatorConstructor(newTrilu, 14, triluTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return triluVersions
}
