package gemm

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gemmVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(
		newGemm9,
		7,
		[][]tensor.Dtype{
			{tensor.Float32, tensor.Float64},
			{tensor.Float32, tensor.Float64},
			{tensor.Float32, tensor.Float64},
		},
	),
	9:  ops.NewOperatorConstructor(newGemm9, 9, gemmTypeConstraints),
	11: ops.NewOperatorConstructor(newGemm, 11, gemmTypeConstraints),
	13: ops.NewOperatorConstructor(newGemm, 13, gemmTypeConstraints),
}

func GetGemmVersions() ops.OperatorVersions {
	return gemmVersions
}
