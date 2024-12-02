package div

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var DivVersions = ops.OperatorVersions{
	7:  newDiv7, // Same, but float16 type differs
	13: newDiv13,
}
