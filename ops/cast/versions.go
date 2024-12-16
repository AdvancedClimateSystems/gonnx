package cast

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var castVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newCast, 6, castTypeConstraints),
	9:  ops.NewOperatorConstructor(newCast, 9, castTypeConstraints),
	13: ops.NewOperatorConstructor(newCast, 13, castTypeConstraints),
}

func GetCastVersions() ops.OperatorVersions {
	return castVersions
}
