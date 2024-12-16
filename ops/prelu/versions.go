package prelu

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var preluVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newPRelu, 7, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}),
	9: ops.NewOperatorConstructor(newPRelu, 9, preluTypeConstraints),
}

func GetPReluVersions() ops.OperatorVersions {
	return preluVersions
}
