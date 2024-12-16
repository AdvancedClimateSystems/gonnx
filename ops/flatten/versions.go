package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var flattenVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newFlatten, 1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}}),
	9:  ops.NewOperatorConstructor(newFlatten, 9, [][]tensor.Dtype{ops.AllTypes}),
	11: ops.NewOperatorConstructor(newFlatten, 11, [][]tensor.Dtype{ops.AllTypes}),
	13: ops.NewOperatorConstructor(newFlatten, 13, [][]tensor.Dtype{ops.AllTypes}),
}

func GetFlattenVersions() ops.OperatorVersions {
	return flattenVersions
}
