package less

import "github.com/advancedclimatesystems/gonnx/ops"

var LessVersions = ops.OperatorVersions{
	7:  newLess7, // Only float types
	9:  newLess9, // bfloat16 type
	13: newLess13,
}
