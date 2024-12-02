package softmax

import "github.com/advancedclimatesystems/gonnx/ops"

var SoftmaxVersions = ops.OperatorVersions{
	1:  newSoftmax1,  // Only support negative dimensions differs
	11: newSoftmax11, // Different default value and bfloat16 type
	13: newSoftmax13,
}
