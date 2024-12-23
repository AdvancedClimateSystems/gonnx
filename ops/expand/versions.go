package expand

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var expandVersions = ops.OperatorVersions{
	8:  ops.NewOperatorConstructor(newExpand, 8, expandTypeConstraints),
	13: ops.NewOperatorConstructor(newExpand, 13, expandTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return expandVersions
}
