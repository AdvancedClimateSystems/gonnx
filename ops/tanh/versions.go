package tanh

import "github.com/advancedclimatesystems/gonnx/ops"

var TanhVersions = ops.OperatorVersions{
	6:  newTanh6, // Only bfloat16 type differs
	13: newTanh13,
}
