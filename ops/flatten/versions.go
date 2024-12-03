package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var FlattenVersions = ops.OperatorVersions{
	1:  newConstructor(newFlatten(1, [][]tensor.Dtype{{tensor.Float32, tensor.Float64}})),
	9:  newConstructor(newFlatten(9, [][]tensor.Dtype{ops.AllTypes})),
	11: newConstructor(newFlatten(11, [][]tensor.Dtype{ops.AllTypes})),
	13: newConstructor(newFlatten(13, [][]tensor.Dtype{ops.AllTypes})),
}

func newConstructor(base *Flatten) func() ops.Operator {
	return func() ops.Operator {
		return base
	}
}
