package gather

import "github.com/advancedclimatesystems/gonnx/ops"

var GatherVersions = ops.OperatorVersions{
	1:  newGather1,  // Same, but with negative axis
	11: newGather11, // Same, but with bfloat16
	13: newGather13,
}
