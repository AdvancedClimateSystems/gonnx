package linearregressor

import "github.com/advancedclimatesystems/gonnx/ops"

var LinearRegressorVersions = ops.OperatorVersions{
	1: newLinearRegressor1,
}
