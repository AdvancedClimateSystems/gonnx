package asin

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AsinVersions = ops.OperatorVersions{
	7: newAsin7,
}
