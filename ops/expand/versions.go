package expand

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ExpandVersions = ops.OperatorVersions{
	8:  newExpand8, // Same, but float16 type differs
	13: newExpand13,
}
