package cast

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var CastVersions = ops.OperatorVersions{
	6:  newCast6, // Same, but string type is added
	9:  newCast9, // Same, but bfloat16 type differs
	13: newCast13,
}
