package reducemax

import "github.com/advancedclimatesystems/gonnx/ops"

var reduceMaxVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newReduceMax, 1, reduceMax11TypeConstraints),
	11: ops.NewOperatorConstructor(newReduceMax, 11, reduceMax11TypeConstraints),
	12: ops.NewOperatorConstructor(newReduceMax, 12, reduceMaxTypeConstraints),
	13: ops.NewOperatorConstructor(newReduceMax, 13, reduceMaxTypeConstraints),
}

func GetReduceMaxVersions() ops.OperatorVersions {
	return reduceMaxVersions
}
