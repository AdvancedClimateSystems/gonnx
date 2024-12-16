package sinh

import "github.com/advancedclimatesystems/gonnx/ops"

var sinhVersions = ops.OperatorVersions{
	9: ops.NewOperatorConstructor(newSinh, 9, sinhTypeConstraints),
}

func GetSinhVersions() ops.OperatorVersions {
	return sinhVersions
}
