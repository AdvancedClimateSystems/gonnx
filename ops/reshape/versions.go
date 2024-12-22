package reshape

import "github.com/advancedclimatesystems/gonnx/ops"

var reshapeVersions = ops.OperatorVersions{
	5:  ops.NewOperatorConstructor(newReshape, 5, reshapeTypeConstraints),
	13: ops.NewOperatorConstructor(newReshape, 13, reshapeTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return reshapeVersions
}
