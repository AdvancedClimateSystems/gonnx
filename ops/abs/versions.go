package abs

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AbsVersions = ops.OperatorVersions{
	6:  newConstructor(newAbs(6, absTypeConstraint)), // Same, but bfloat16 type is added
	13:  newConstructor(newAbs(13, absTypeConstraint)), // Same, but bfloat16 type is added
}

func newConstructor(base *Abs) func() ops.Operator {
	return func() ops.Operator {
		return base
	}
}
