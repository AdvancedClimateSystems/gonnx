package slice

import "github.com/advancedclimatesystems/gonnx/ops"

var SliceVersions = ops.OperatorVersions{
	1:  newSlice1,  // Different attributes and implementation
	10: newSlice10, // Only negative indexing differs
	11: newSlice11, // Only bfloat16 type differs
	13: newSlice13,
}
