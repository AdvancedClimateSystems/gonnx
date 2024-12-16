package tanh

import "github.com/advancedclimatesystems/gonnx/ops"

var tanhVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newTanh, 6, tanhTypeConstraint),
	13: ops.NewOperatorConstructor(newTanh, 13, tanhTypeConstraint),
}

func GetTanhVersions() ops.OperatorVersions {
	return tanhVersions
}
