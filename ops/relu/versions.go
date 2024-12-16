package relu

import "github.com/advancedclimatesystems/gonnx/ops"

var reluVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newRelu, 6, reluTypeConstraints),
	13: ops.NewOperatorConstructor(newRelu, 13, reluTypeConstraints),
}

func GetReluVersions() ops.OperatorVersions {
	return reluVersions
}
