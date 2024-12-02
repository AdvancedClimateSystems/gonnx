package reducemax

import "github.com/advancedclimatesystems/gonnx/ops"

var ReduceMaxVersions = ops.OperatorVersions{
	1:  newReduceMax1,  // Only negative dimensions differ
	11: newReduceMax11, // Only int types differ
	12: newReduceMax12, // Only bfloat16 type differs
	13: newReduceMax13,
}
