package sin

import "github.com/advancedclimatesystems/gonnx/ops"

var SinVersions = ops.OperatorVersions{
	7: newSin7,
}
