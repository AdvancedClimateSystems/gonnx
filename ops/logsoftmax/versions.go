package logsoftmax

import "github.com/advancedclimatesystems/gonnx/ops"

var logSoftmaxVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newLogSoftmax, 1, logSoftmaxTypeConstraints),
	11: ops.NewOperatorConstructor(newLogSoftmax, 11, logSoftmaxTypeConstraints),
	13: ops.NewOperatorConstructor(newLogSoftmax, 13, logSoftmaxTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return logSoftmaxVersions
}
