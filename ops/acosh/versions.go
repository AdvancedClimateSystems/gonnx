package acosh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AcoshVersions = ops.OperatorVersions{
	9: newAcosh9,
}
