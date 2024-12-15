package concat

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var concatVersions = ops.OperatorVersions{
	4:  ops.NewOperatorConstructor(newConcat(4)),
	11: ops.NewOperatorConstructor(newConcat(11)),
	13: ops.NewOperatorConstructor(newConcat(13)),
}

func GetConcatVersions() ops.OperatorVersions {
	return concatVersions
}
