package sin

import "github.com/advancedclimatesystems/gonnx/ops"

var sinVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newSin, 7, sinTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return sinVersions
}
