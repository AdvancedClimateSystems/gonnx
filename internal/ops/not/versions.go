package not

import "github.com/advancedclimatesystems/gonnx/internal/ops"

var notVersions = ops.OperatorVersions{
	1: ops.NewOperatorConstructor(newNot, 1, notTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return notVersions
}
