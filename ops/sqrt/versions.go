package sqrt

import "github.com/advancedclimatesystems/gonnx/ops"

var sqrtVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newSqrt, 6, sqrtTypeConstraints),
	13: ops.NewOperatorConstructor(newSqrt, 13, sqrtTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return sqrtVersions
}
