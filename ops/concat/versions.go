package concat

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var concatVersions = ops.OperatorVersions{
	4:  ops.NewOperatorConstructor(newConcat, 4, concatTypeConstraints),
	11: ops.NewOperatorConstructor(newConcat, 11, concatTypeConstraints),
	13: ops.NewOperatorConstructor(newConcat, 13, concatTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return concatVersions
}
