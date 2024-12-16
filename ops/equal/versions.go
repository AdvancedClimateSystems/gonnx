package equal

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var equalVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newEqual, 7, equal7TypeConstraints),
	11: ops.NewOperatorConstructor(newEqual, 11, equalTypeConstraints),
	13: ops.NewOperatorConstructor(newEqual, 13, equalTypeConstraints),
}

func GetEqualVersions() ops.OperatorVersions {
	return equalVersions
}
