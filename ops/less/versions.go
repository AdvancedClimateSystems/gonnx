package less

import "github.com/advancedclimatesystems/gonnx/ops"

var lessVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newLess, 7, less7TypeConstraints),
	9:  ops.NewOperatorConstructor(newLess, 9, lessTypeConstraints),
	13: ops.NewOperatorConstructor(newLess, 13, lessTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return lessVersions
}
