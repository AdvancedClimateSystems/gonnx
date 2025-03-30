package gru

import "github.com/advancedclimatesystems/gonnx/ops"

var gruVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newGRU, 7, gruTypeConstraints),
	14: ops.NewOperatorConstructor(newGRU, 14, gruTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return gruVersions
}
