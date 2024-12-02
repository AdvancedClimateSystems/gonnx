package abs

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AbsVersions = ops.OperatorVersions{
	6:  newAbs6, // Same, but bfloat16 type is added
	13: newAbs13,
}
