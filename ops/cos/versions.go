package cos

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var CosVersions = ops.OperatorVersions{
	7: newCos7,
}
