package constant

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var constantVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newConstant9, 1, [][]tensor.Dtype{}),
	9:  ops.NewOperatorConstructor(newConstant9, 9, [][]tensor.Dtype{}),
	11: newConstant11,
	12: ops.NewOperatorConstructor(newConstant, 12, [][]tensor.Dtype{}),
	13: ops.NewOperatorConstructor(newConstant, 13, [][]tensor.Dtype{}),
}

func GetVersions() ops.OperatorVersions {
	return constantVersions
}
