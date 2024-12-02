package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var FlattenVersions = ops.OperatorVersions{
	1:  newFlatten1,
	9:  newFlatten9,
	11: newFlatten11,
	13: newFlatten13,
}
