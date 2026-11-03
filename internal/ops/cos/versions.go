package cos

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
)

var cosVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newCos, 7, cosTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return cosVersions
}
