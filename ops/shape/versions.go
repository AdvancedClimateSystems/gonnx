package shape

import "github.com/advancedclimatesystems/gonnx/ops"

var ShapeVersions = ops.OperatorVersions{
	1:  newShape1, // Only bfloat16 type differs
	13: newShape13,
}
