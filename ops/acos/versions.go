package acos

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AcosVersions = ops.OperatorVersions{
	7: newAcos7,
}
