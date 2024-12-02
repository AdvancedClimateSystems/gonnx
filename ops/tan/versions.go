package tan

import "github.com/advancedclimatesystems/gonnx/ops"

var TanVersions = ops.OperatorVersions{
	7: newTan7,
}
