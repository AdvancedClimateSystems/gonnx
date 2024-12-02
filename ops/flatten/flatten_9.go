package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Flatten9 implements version-specific behavior for Flatten 9.
type Flatten9 struct {
	*FlattenBase
}

func newFlatten9() ops.Operator {
	return &Flatten9{
		FlattenBase: &FlattenBase{
			version:   9,
			axis:      1,
			minInputs: 1,
			maxInputs: 1,
			// Misses types float16, string, complex64, complex128
			inputTypeConstraints: [][]tensor.Dtype{ops.AllTypes},
		},
	}
}
