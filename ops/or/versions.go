package or

import "github.com/advancedclimatesystems/gonnx/ops"

var orVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newOr, 7, orTypeConstraints),
}

func GetOrVersions() ops.OperatorVersions {
	return orVersions
}
