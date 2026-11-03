package lstm

import "github.com/advancedclimatesystems/gonnx/internal/ops"

var lstmVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newLSTM, 7, lstmTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return lstmVersions
}
