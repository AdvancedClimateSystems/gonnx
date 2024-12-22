package scaler

import "github.com/advancedclimatesystems/gonnx/ops"

var scalerVersions = ops.OperatorVersions{
	1: ops.NewOperatorConstructor(newScaler, 1, scalerTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return scalerVersions
}
