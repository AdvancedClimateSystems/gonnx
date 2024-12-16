package constant

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var constantVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newConstantLegacy(1)),
	9:  ops.NewOperatorConstructor(newConstantLegacy(9)),
	11: ops.NewOperatorConstructor(newConstant11()),
	12: ops.NewOperatorConstructor(newConstant(12)),
	13: ops.NewOperatorConstructor(newConstant(13)),
}

func GetConstantVersions() ops.OperatorVersions {
	return constantVersions
}
