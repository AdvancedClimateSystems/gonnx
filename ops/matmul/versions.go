package matmul

import "github.com/advancedclimatesystems/gonnx/ops"

var matMulVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newMatMul, 1, matmul1TypeConstraints),
	9:  ops.NewOperatorConstructor(newMatMul, 9, matmulTypeConstraints),
	13: ops.NewOperatorConstructor(newMatMul, 13, matmulTypeConstraints),
}

func GetMatMulVersions() ops.OperatorVersions {
	return matMulVersions
}
