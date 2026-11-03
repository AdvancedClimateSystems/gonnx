package constant

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var constantVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newConstant9, 1, [][]tensor.DataType{}),
	9:  ops.NewOperatorConstructor(newConstant9, 9, [][]tensor.DataType{}),
	11: newConstant11,
	12: ops.NewOperatorConstructor(newConstant, 12, [][]tensor.DataType{}),
	13: ops.NewOperatorConstructor(newConstant, 13, [][]tensor.DataType{}),
}

func GetVersions() ops.OperatorVersions {
	return constantVersions
}
