package ops

// UnknownAttributeErrTemplate is used to format an error
// when an operator finds an unknown attribute during its initialization.
const UnknownAttributeErrTemplate = "%v: unknown attribute: %v"

// UnsupportedAttrErrTemplate is used to format an error when an operator receives
// an attribute that is not supported yet.
const UnsupportedAttrErrTemplate = "%v: %v attribute not supported yet"

// InvalidAttrTemplate is used to format an error when a known attribute could not
// be parsed/interpreted correctly.
const InvalidAttrTemplate = "%v: attribute %v could not be parsed as %T"

// InvalidAttrCountErrTemplate is used to format an error when an operator
// got the wrong amount of attributes.
const InvalidAttrCountErrTemplate = "%v: expected %v attributes, got %d"

// InvalidInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors.
const InvalidInputCountErrTemplate = "%v: expected %d input tensors, got %d"

// InvalidOptionalInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors when optional inputs are present.
const InvalidOptionalInputCountErrTemplate = "%v: expected %d-%d input tensors, got %d"

// UnknowOpTypeErrTemplate is used to format an error when the operator type is unknown.
const UnknowOpTypeErrTemplate = "unknown operator type: %v"

// MultidirBroadcastErrTemplate is used to format an error when two inputs cannot be
// broadcasted together with Multidirectional broadcasting.
const MultidirBroadcastErrTemplate = "could not multidir broadcast inputs with shape %d and %d: %v"

// UnidirBroadcastErrTemplate is used to format an error when two inputs cannot be
// broadcasted together with Unidirectional broadcasting.
const UnidirBroadcastErrTemplate = "could not unidir broadcast inputs with shape %d and %d"

// AxisOutOfRangeErrTemplate is used to format an error when an given axis is out of range
// given a certain rank.
const AxisOutOfRangeErrTemplate = "axis argument must be in the range -%d <= x < %d, was %d"

// AxesNotAllInRangeErrTemplate is used to format an error when not all indices
// are within a given range.
const AxesNotAllInRangeErrTemplate = "all indices entries must be in the range -%d <= x < %d"
