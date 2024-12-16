package conv

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var convVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newConv, 1, convTypeConstraints),
	11: ops.NewOperatorConstructor(newConv, 11, convTypeConstraints),
}

func GetConvVersions() ops.OperatorVersions {
	return convVersions
}
