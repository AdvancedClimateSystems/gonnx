package squeeze

import "github.com/advancedclimatesystems/gonnx/ops"

var squeezeVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newSqueeze1()),
	11: ops.NewOperatorConstructor(newSqueeze11()),
	13: ops.NewOperatorConstructor(newSqueeze(13, squeezeTypeConstraints)),
}

func GetSqueezeVersions() ops.OperatorVersions {
	return squeezeVersions
}
