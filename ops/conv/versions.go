package conv

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ConvVersions = ops.OperatorVersions{
	1:  newConv1, // Same, but only float16 type differs
	11: newConv11,
}
