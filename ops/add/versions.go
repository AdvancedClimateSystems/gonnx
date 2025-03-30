package add

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var addVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newAdd, 7, addTypeConstraints),
	13: ops.NewOperatorConstructor(newAdd, 13, addTypeConstraints),
	14: ops.NewOperatorConstructor(newAdd, 14, add14TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return addVersions
}

var add14TypeConstraints = [][]tensor.Dtype{
	ops.NumericTypes,
	ops.NumericTypes,
}
