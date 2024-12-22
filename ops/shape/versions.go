package shape

import "github.com/advancedclimatesystems/gonnx/ops"

var shapeVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newShape, 1, shapeTypeConstraints),
	13: ops.NewOperatorConstructor(newShape, 13, shapeTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return shapeVersions
}
