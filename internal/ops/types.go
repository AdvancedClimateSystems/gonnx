package ops

import "github.com/advancedclimatesystems/gonnx/tensor"

// FloatType is a type that describes a float value. Can be either float32 or float64.
type FloatType = tensor.Float

// IntType is a type that describes an integer value.
type IntType = tensor.Integer

// NumericType is a type that describes a numeric value.
type NumericType = tensor.Numeric

// AllTypes is a type constraint which allows all types.
var AllTypes = []tensor.DataType{
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Float32, tensor.Float64,
	tensor.Bool,
}

// IntTypes is a list with all integer types.
var IntTypes = []tensor.DataType{
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
}

// NumericTypes is a list with all numeric types.
var NumericTypes = []tensor.DataType{
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Float32, tensor.Float64,
}
