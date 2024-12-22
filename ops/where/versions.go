package where

import "github.com/advancedclimatesystems/gonnx/ops"

var whereVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newWhere, 9, whereTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return whereVersions
}
