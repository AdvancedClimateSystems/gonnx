package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var FlattenVersions = ops.OperatorVersions{
	1:  newFlatten1,  // Same, but only float types
	9:  newFlatten9,  // Same, but negative axis added
	11: newFlatten11, // Same, but float16 type differs
	13: newFlatten13,
}
