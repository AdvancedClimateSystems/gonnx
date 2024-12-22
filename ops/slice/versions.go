package slice

import "github.com/advancedclimatesystems/gonnx/ops"

var sliceVersions = ops.OperatorVersions{
	1:  newSlice1,
	10: ops.NewOperatorConstructor(newSlice, 10, sliceTypeConstraints),
	11: ops.NewOperatorConstructor(newSlice, 11, sliceTypeConstraints),
	13: ops.NewOperatorConstructor(newSlice, 13, sliceTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return sliceVersions
}
