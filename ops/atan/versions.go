package atan

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AtanVersions = ops.OperatorVersions{
	7: newAtan7,
}
