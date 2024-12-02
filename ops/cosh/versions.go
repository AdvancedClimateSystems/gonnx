package cosh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var CoshVersions = ops.OperatorVersions{
	9: newCosh9,
}
