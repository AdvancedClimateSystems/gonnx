package tan

import "github.com/advancedclimatesystems/gonnx/ops"

var TanVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newTan(7, tanTypeConstraints)),
}
