package prelu

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var preluVersions = ops.OperatorVersions{
	7: ops.NewOperatorConstructor(newPRelu, 7, [][]tensor.DataType{{tensor.Float32, tensor.Float64}, {tensor.Float32, tensor.Float64}}),
	9: ops.NewOperatorConstructor(newPRelu, 9, preluTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return preluVersions
}
