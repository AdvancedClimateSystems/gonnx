package lessorequal

import "github.com/advancedclimatesystems/gonnx/ops"

var LessOrEqualVersions = ops.OperatorVersions{
	12: newLessOrEqual12,
}
