package div

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var divVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newDiv, 7, divTypeConstraints),
	13: ops.NewOperatorConstructor(newDiv, 13, divTypeConstraints),
	14: ops.NewOperatorConstructor(newDiv, 14, div14TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return divVersions
}

var div14TypeConstraints = [][]tensor.Dtype{
	ops.NumericTypes,
	ops.NumericTypes,
}
