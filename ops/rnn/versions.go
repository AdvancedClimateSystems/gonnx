package rnn

import "github.com/advancedclimatesystems/gonnx/ops"

var rnnVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newRNN, 7, rnnTypeConstraints),
}

func GetRNNVersions() ops.OperatorVersions {
	return rnnVersions
}
