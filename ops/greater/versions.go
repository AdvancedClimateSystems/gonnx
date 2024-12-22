package greater

import "github.com/advancedclimatesystems/gonnx/ops"

var greaterVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newGreater, 7, greater7TypeConstraints),
	9:  ops.NewOperatorConstructor(newGreater, 9, greaterTypeConstraints),
	13: ops.NewOperatorConstructor(newGreater, 13, greaterTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return greaterVersions
}
