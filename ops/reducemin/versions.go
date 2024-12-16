package reducemin

import "github.com/advancedclimatesystems/gonnx/ops"

var reduceMinVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newReduceMin, 1, reduceMin11TypeConstraints),
	11: ops.NewOperatorConstructor(newReduceMin, 11, reduceMin11TypeConstraints),
	12: ops.NewOperatorConstructor(newReduceMin, 12, reduceMinTypeConstraints),
	13: ops.NewOperatorConstructor(newReduceMin, 13, reduceMinTypeConstraints),
}

func GetReduceMinVersions() ops.OperatorVersions {
	return reduceMinVersions
}
