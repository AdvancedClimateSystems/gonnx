package reshape

import "github.com/advancedclimatesystems/gonnx/ops"

var ReshapeVersions = ops.OperatorVersions{
	5:  newReshape5, // Only bfloat16 type differs
	13: newReshape13,
}
