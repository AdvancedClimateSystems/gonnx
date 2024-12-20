package reducemean

import "github.com/advancedclimatesystems/gonnx/ops"

var reduceMeanVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newReduceMean, 1, reduceMeanTypeConstraints),
	11: ops.NewOperatorConstructor(newReduceMean, 11, reduceMeanTypeConstraints),
	13: ops.NewOperatorConstructor(newReduceMean, 13, reduceMeanTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return reduceMeanVersions
}
