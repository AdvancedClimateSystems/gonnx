package unsqueeze

import "github.com/advancedclimatesystems/gonnx/ops"

var UnsqueezeVersions = ops.OperatorVersions{
	1:  newUnsqueeze1,  // Only negative dimensions differs
	11: newUnsqueeze11, // Switched input to attr and bfloat16 type differs
	13: newUnsqueeze13,
}
