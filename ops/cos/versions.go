package cos

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var cosVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newCos, 7, cosTypeConstraints),
}

func GetCosVersions() ops.OperatorVersions {
	return cosVersions
}
