package reducemin

import "github.com/advancedclimatesystems/gonnx/ops"

var ReduceMinVersions = ops.OperatorVersions{
	1:  newReduceMin1,  // Only negative dimensions differ
	11: newReduceMin11, // Only int types differ
	12: newReduceMin12, // Only bfloat16 type differs
	13: newReduceMin13,
}
