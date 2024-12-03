package unsqueeze

import "github.com/advancedclimatesystems/gonnx/ops"

var UnsqueezeVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newUnsqueeze1()),
	11: ops.NewOperatorConstructor(newUnsqueeze11()),
	13: ops.NewOperatorConstructor(newUnsqueeze(13, unsqueezeTypeConstraints)),
}
