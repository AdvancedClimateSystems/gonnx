package ops

import "gorgonia.org/tensor"

type FloatType interface {
	float32 | float64
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
