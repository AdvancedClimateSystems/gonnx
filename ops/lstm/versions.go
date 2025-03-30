package lstm

import "github.com/advancedclimatesystems/gonnx/ops"

var lstmVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newLSTM, 7, lstmTypeConstraints),
	14: ops.NewOperatorConstructor(newLSTM, 14, lstmTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return lstmVersions
}
