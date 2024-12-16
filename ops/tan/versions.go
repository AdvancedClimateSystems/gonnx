package tan

import "github.com/advancedclimatesystems/gonnx/ops"

var tanVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newTan, 7, tanTypeConstraints),
}

func GetTanVersions() ops.OperatorVersions {
	return tanVersions
}
