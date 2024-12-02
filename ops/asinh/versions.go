package asinh

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AsinhVersions = ops.OperatorVersions{
	9: newAsinh9,
}
