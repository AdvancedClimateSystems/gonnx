package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Flatten1 implements version-specific behavior for Flatten 1.
type Flatten1 struct {
	*FlattenBase
}

func newFlatten1() ops.Operator {
	return &Flatten1{
		FlattenBase: &FlattenBase{
			version:   1,
			axis:      1,
			minInputs: 1,
			maxInputs: 1,
			// Misses Float16 type
			inputTypeConstraints: [][]tensor.Dtype{{tensor.Float32, tensor.Float64}},
		},
	}
}
