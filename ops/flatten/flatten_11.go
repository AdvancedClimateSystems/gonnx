package flatten

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Flatten11 implements version-specific behavior for Flatten 11.
type Flatten11 struct {
	*FlattenBase
}

func newFlatten11() ops.Operator {
	return &Flatten11{
		FlattenBase: &FlattenBase{
			version:   11,
			axis:      1,
			minInputs: 1,
			maxInputs: 1,
			// Misses types float16, string, complex64, complex128
			inputTypeConstraints: [][]tensor.Dtype{ops.AllTypes},
		},
	}
}
