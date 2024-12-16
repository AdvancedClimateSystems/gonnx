package add

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var addVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newAdd, 7, addTypeConstraints),
	13: ops.NewOperatorConstructor(newAdd, 13, addTypeConstraints),
}

func GetAddVersions() ops.OperatorVersions {
	return addVersions
}
