package sigmoid

import "github.com/advancedclimatesystems/gonnx/ops"

var sigmoidVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newSigmoid, 6, sigmoidTypeConstraints),
	13: ops.NewOperatorConstructor(newSigmoid, 13, sigmoidTypeConstraints),
}

func GetSigmoidVersions() ops.OperatorVersions {
	return sigmoidVersions
}
