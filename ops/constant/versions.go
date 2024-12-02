package constant

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var ConstantVersions = ops.OperatorVersions{
	1:  newConstant1,
	9:  newConstant9,
	11: newConstant11,
	12: newConstant12, // Same, but bfloat16 type differs
	13: newConstant13,
}
