package squeeze

import "github.com/advancedclimatesystems/gonnx/ops"

var SqueezeVersions = ops.OperatorVersions{
	1:  newSqueeze1,  // Supports negative dimensions as only difference
	11: newSqueeze11, // Switch from input to attribute, implementation fairly same
	13: newSqueeze13,
}
