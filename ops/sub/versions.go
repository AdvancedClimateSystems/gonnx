package sub

import "github.com/advancedclimatesystems/gonnx/ops"

var SubVersions = ops.OperatorVersions{
	7:  newSub7,
	13: newSub13,
}
