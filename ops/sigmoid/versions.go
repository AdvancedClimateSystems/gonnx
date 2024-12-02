package sigmoid

import "github.com/advancedclimatesystems/gonnx/ops"

var SigmoidVersions = ops.OperatorVersions{
	6:  newSigmoid6, // Only bfloat16 type differs
	13: newSigmoid13,
}
