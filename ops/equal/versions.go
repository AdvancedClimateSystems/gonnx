package equal

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var EqualVersions = ops.OperatorVersions{
	7:  newEqual7,
	11: newEqual11, // Same, but float16 type differs
	13: newEqual13,
}
