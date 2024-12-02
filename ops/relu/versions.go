package relu

import "github.com/advancedclimatesystems/gonnx/ops"

var ReluVersions = ops.OperatorVersions{
	6:  newRelu6, // Only bfloat16 type differs
	13: newRelu13,
}
