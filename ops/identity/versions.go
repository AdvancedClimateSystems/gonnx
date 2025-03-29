package identity

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var identityVersions = ops.OperatorVersions{
	13: ops.NewOperatorConstructor(newIdentity, 13, identityTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return identityVersions
}
