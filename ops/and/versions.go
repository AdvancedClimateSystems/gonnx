package and

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var andVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newAnd, 7, andTypeConstraints),
}

func GetAndVersions() ops.OperatorVersions {
	return andVersions
}
