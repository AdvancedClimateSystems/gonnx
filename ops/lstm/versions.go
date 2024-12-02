package lstm

import "github.com/advancedclimatesystems/gonnx/ops"

var LSTMVersions = ops.OperatorVersions{
	7: newLSTM7,
}
