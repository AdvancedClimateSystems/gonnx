package mul

import "github.com/advancedclimatesystems/gonnx/ops"

var MulVersions = ops.OperatorVersions{
	7:  newMul7, // Only bfloat16 type differs
	13: newMul13,
}
