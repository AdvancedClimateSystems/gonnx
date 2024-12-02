package gemm

import "github.com/advancedclimatesystems/gonnx/ops"

var GemmVersions = ops.OperatorVersions{
	7:  newGemm7,
	9:  newGemm9,
	11: newGemm11,
	13: newGemm13,
}
