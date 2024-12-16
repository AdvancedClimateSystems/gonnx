package asinh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var asinhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newAsinh, 9, asinhTypeConstraints),
}

func GetAsinhVersions() ops.OperatorVersions {
	return asinhVersions
}
