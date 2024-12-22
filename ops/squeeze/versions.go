package squeeze

import "github.com/advancedclimatesystems/gonnx/ops"

var squeezeVersions = ops.OperatorVersions{
	1:  newSqueeze1,
	11: newSqueeze11,
	13: ops.NewOperatorConstructor(newSqueeze, 13, squeezeTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return squeezeVersions
}
