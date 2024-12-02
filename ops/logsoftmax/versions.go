package logsoftmax

import "github.com/advancedclimatesystems/gonnx/ops"

var LogSoftmaxVersions = ops.OperatorVersions{
	1:  newLogSoftmax1,  // Only adds negative dimension support
	11: newLogSoftmax11, // Only bfloat16 type differs and default differs
	13: newLogSoftmax13,
}
