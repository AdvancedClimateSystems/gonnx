package lessorequal

import "github.com/advancedclimatesystems/gonnx/ops"

var lessOrEqualVersions = ops.OperatorVersions{
	12: ops.NewOperatorConstructor(newLessOrEqual, 12, lessOrEqualTypeConstraints),
}

func GetLessOrEqualVersions() ops.OperatorVersions {
	return lessOrEqualVersions
}
