package greater

import "github.com/advancedclimatesystems/gonnx/ops"

var GreaterVersions = ops.OperatorVersions{
	7:  newGreater7,
	9:  newGreater9,
	13: newGreater13,
}
