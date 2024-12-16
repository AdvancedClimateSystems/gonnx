package div

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var divVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newDiv, 7, divTypeConstraints),
	13: ops.NewOperatorConstructor(newDiv, 13, divTypeConstraints),
}

func GetDivVersions() ops.OperatorVersions {
	return divVersions
}
