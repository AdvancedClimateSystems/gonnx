package asin

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
)

var asinVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newAsin, 7, asinTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return asinVersions
}
