package lstm

import "github.com/advancedclimatesystems/gonnx/ops"

var lstmVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newLSTM, 7, lstmTypeConstraints),
}

func GetLSTMVersions() ops.OperatorVersions {
	return lstmVersions
}
