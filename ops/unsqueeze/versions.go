package unsqueeze

import "github.com/advancedclimatesystems/gonnx/ops"

var unsqueezeVersions = ops.OperatorVersions{
	1:  newUnsqueeze1,
	11: newUnsqueeze11,
	13: ops.NewOperatorConstructor(newUnsqueeze, 13, unsqueezeTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return unsqueezeVersions
}
