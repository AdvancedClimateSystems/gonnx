package flatten

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var flattenVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newFlatten, 1, [][]tensor.DataType{{tensor.Float32, tensor.Float64}}),
	9:  ops.NewOperatorConstructor(newFlatten, 9, [][]tensor.DataType{ops.AllTypes}),
	11: ops.NewOperatorConstructor(newFlatten, 11, [][]tensor.DataType{ops.AllTypes}),
	13: ops.NewOperatorConstructor(newFlatten, 13, [][]tensor.DataType{ops.AllTypes}),
}

func GetVersions() ops.OperatorVersions {
	return flattenVersions
}
