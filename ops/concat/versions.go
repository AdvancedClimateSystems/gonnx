package concat

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ConcatVersions = ops.OperatorVersions{
	4:  newConcat4,
	11: newConcat11, // Same, but bfloat16 type differs
	13: newConcat13,
}
