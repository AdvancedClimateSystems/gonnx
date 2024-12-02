package greaterorequal

import "github.com/advancedclimatesystems/gonnx/ops"

var GreaterOrEqualVersions = ops.OperatorVersions{
	12: newGreaterOrEqual12,
}
