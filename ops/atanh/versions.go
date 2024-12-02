package atanh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AtanhVersions = ops.OperatorVersions{
	9: newAtanh9,
}
