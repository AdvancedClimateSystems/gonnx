package matmul

import "github.com/advancedclimatesystems/gonnx/ops"

var MatMulVersions = ops.OperatorVersions{
	1:  newMatMul1, // Only float types
	9:  newMatMul9, // Only bfloat16 type differs
	13: newMatMul13,
}
