package gru

import "github.com/advancedclimatesystems/gonnx/ops"

var gruVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newGRU, 7, gruTypeConstraints),
}

func GetGRUVersions() ops.OperatorVersions {
	return gruVersions
}
