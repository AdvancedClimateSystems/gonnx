package softmax

import "github.com/advancedclimatesystems/gonnx/ops"

var softmaxVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newSoftmax, 1, softmaxTypeConstraints),
	11: ops.NewOperatorConstructor(newSoftmax, 11, softmaxTypeConstraints),
	13: ops.NewOperatorConstructor(newSoftmax, 13, softmaxTypeConstraints),
}

func GetSoftmaxVersions() ops.OperatorVersions {
	return softmaxVersions
}
