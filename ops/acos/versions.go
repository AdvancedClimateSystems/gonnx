package acos

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AcosVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newAcos(7)),
}
