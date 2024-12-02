package scaler

import "github.com/advancedclimatesystems/gonnx/ops"

var ScalerVersions = ops.OperatorVersions{
	1: newScaler1,
}
