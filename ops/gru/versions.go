package gru

import "github.com/advancedclimatesystems/gonnx/ops"

var GRUVersions = ops.OperatorVersions{
	7: newGRU7,
}
