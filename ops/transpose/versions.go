package transpose

import "github.com/advancedclimatesystems/gonnx/ops"

var TransposeVersions = ops.OperatorVersions{
	1:  ops.NewOperatorConstructor(newTranspose(1, transposeTypeConstraint)),
	13: ops.NewOperatorConstructor(newTranspose(13, transposeTypeConstraint)),
}
