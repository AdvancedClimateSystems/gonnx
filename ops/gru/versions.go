package gru

import "github.com/advancedclimatesystems/gonnx/ops"

var gruVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newGRU, 7, gruTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return gruVersions
}
