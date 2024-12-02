package greater

import "github.com/advancedclimatesystems/gonnx/ops"

var GreaterVersions = ops.OperatorVersions{
	7:  newGreater7, // Only float types
	9:  newGreater9, // bfloat16 added
	13: newGreater13,
}
