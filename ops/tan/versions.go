package tan

import "github.com/advancedclimatesystems/gonnx/ops"

var tanVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newTan, 7, tanTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return tanVersions
}
