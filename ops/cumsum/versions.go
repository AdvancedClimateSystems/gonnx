package cumsum

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var cumsumVersions = ops.OperatorVersions{
	11: ops.NewOperatorConstructor(newCumSum, 11, cumsumTypeConstraints),
	14: ops.NewOperatorConstructor(newCumSum, 14, cumsumTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return cumsumVersions
}
