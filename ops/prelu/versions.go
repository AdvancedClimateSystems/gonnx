package prelu

import "github.com/advancedclimatesystems/gonnx/ops"

var PReluVersions = ops.OperatorVersions{
	7: newPRelu7, // Only float types
	9: newPRelu9,
}
