package relu

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reluVersions = ops.OperatorVersions{
	6:  ops.NewOperatorConstructor(newRelu, 6, reluTypeConstraints),
	13: ops.NewOperatorConstructor(newRelu, 13, reluTypeConstraints),
	14: ops.NewOperatorConstructor(newRelu, 14, relu14TypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return reluVersions
}

var relu14TypeConstraints = [][]tensor.Dtype{{tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64}}
