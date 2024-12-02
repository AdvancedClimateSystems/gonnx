package rnn

import "github.com/advancedclimatesystems/gonnx/ops"

var RNNVersions = ops.OperatorVersions{
	7: newRNN7,
}
