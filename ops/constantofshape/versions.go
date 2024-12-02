package constantofshape

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ConstantOfShapeVersions = ops.OperatorVersions{
	9: newConstantOfShape9,
}
