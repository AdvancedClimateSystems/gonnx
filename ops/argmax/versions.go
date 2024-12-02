package argmax

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ArgMaxVersions = ops.OperatorVersions{
	11: newArgMax11, // Same, but one attribute is added (which we don't support it anyway)
	12: newArgMax12, // Same, but bfloat16 type differs
	13: newArgMax13,
}
