package sub

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var subVersions = ops.OperatorVersions{
	7:  ops.NewOperatorConstructor(newSub, 7, subTypeConstraints),
	13: ops.NewOperatorConstructor(newSub, 13, subTypeConstraints),
	14: ops.NewOperatorConstructor(newSub, 14, sub14TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return subVersions
}

var sub14TypeConstraints = [][]tensor.Dtype{
	ops.NumericTypes,
	ops.NumericTypes,
}
