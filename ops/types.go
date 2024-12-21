package ops

import "gorgonia.org/tensor"

// FloatType is a type that describes a float value. Can be either float32 or float64.
type FloatType interface {
	float32 | float64
}

type NumericType interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | FloatType
}

// AllTypes is a type constraint which allows all types.
var AllTypes = []tensor.Dtype{
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Float32, tensor.Float64,
	tensor.Complex64, tensor.Complex128,
	tensor.String,
	tensor.Bool,
}

// NumericTypes is a list with all numeric types.
var NumericTypes = []tensor.Dtype{
	tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64,
	tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
	tensor.Float32, tensor.Float64,
}
