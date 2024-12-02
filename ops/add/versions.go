package add

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var AddVersions = ops.OperatorVersions{
	7:  newAdd7, // Same, but bfloat16 type is added
	13: newAdd13,
}
