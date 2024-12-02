package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Flatten13 implements version-specific behavior for Flatten 13.
type Flatten13 struct {
	*FlattenBase
}

func newFlatten13() ops.Operator {
	return &Flatten13{
		FlattenBase: &FlattenBase{
			version:   13,
			axis:      1,
			minInputs: 1,
			maxInputs: 1,
			// Misses types float16, bfloat16, string, complex64, complex128
			inputTypeConstraints: [][]tensor.Dtype{ops.AllTypes},
		},
	}
}
