package identity

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
)

var identityVersions = ops.OperatorVersions{
	13: ops.NewOperatorConstructor(newIdentity, 13, identityTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return identityVersions
}
