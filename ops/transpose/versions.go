package transpose

import "github.com/advancedclimatesystems/gonnx/ops"

var TransposeVersions = ops.OperatorVersions{
	1:  newTranspose1, // Only bfloat16 type differs
	13: newTranspose13,
}
