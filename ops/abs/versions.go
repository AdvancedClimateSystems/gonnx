package abs

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AbsVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newAbs(6, absTypeConstraint)),  // Same, but bfloat16 type is added
	13: ops.NewOperatorConstructor(newAbs(13, absTypeConstraint)), // Same, but bfloat16 type is added
}
