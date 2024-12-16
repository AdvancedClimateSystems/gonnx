package constantofshape

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var constantOfShapeVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newConstantOfShape, 9, constantOfShapeTypeConstraints),
}

func GetConstantOfShapeVersions() ops.OperatorVersions {
	return constantOfShapeVersions
}
