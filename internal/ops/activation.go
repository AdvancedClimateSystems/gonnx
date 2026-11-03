package ops

import (
	"math"
	"unsafe"

	"github.com/advancedclimatesystems/gonnx/tensor"
)

// Activation is an element-wise activation function of the recurrent operators LSTM, GRU
// and RNN.
type Activation int

// The supported activation functions.
const (
	ActivationTanh Activation = iota
	ActivationSigmoid
	ActivationReLU
)

// ParseActivation returns the activation function with the given ONNX name.
func ParseActivation(name string) (Activation, error) {
	switch name {
	case "tanh":
		return ActivationTanh, nil
	case "sigmoid":
		return ActivationSigmoid, nil
	case "relu":
		return ActivationReLU, nil
	default:
		return 0, ErrActivationNotImplemented(name)
	}
}

// Activate applies an activation function to every value in place.
func Activate[T FloatType](values []T, activation Activation) {
	switch activation {
	case ActivationTanh:
		for i, value := range values {
			values[i] = tanh(value)
		}
	case ActivationSigmoid:
		for i, value := range values {
			values[i] = sigmoid(value)
		}
	case ActivationReLU:
		for i, value := range values {
			values[i] = relu(value)
		}
	}
}

// Tanh performs the tanh operation on a tensor.
func Tanh(X *tensor.Tensor) (*tensor.Tensor, error) {
	return mapFloat(X, tanh[float32], tanh[float64])
}

// Sigmoid performs the sigmoid operation on a tensor.
func Sigmoid(X *tensor.Tensor) (*tensor.Tensor, error) {
	return mapFloat(X, sigmoid[float32], sigmoid[float64])
}

// ReLU performs the ReLU operation on a tensor.
func ReLU(X *tensor.Tensor) (*tensor.Tensor, error) {
	return mapFloat(X, relu[float32], relu[float64])
}

func mapFloat(X *tensor.Tensor, f32 func(float32) float32, f64 func(float64) float64) (*tensor.Tensor, error) {
	switch X.DataType() {
	case tensor.Float32:
		return Map(X, f32), nil
	case tensor.Float64:
		return Map(X, f64), nil
	default:
		return nil, ErrInvalidType(X.DataType())
	}
}

// tanh and sigmoid compute in float32 for float32 values: the result is at most 2 ulp
// from the correctly rounded value, at a fraction of the cost of float64 math.
func tanh[T FloatType](x T) T {
	if isFloat32[T]() {
		return T(tanh32(float32(x)))
	}

	return T(math.Tanh(float64(x)))
}

func sigmoid[T FloatType](x T) T {
	if isFloat32[T]() {
		return T(sigmoid32(float32(x)))
	}

	return T(1 / (1 + math.Exp(-float64(x))))
}

// isFloat32 reports whether T is float32 rather than float64. Generic code is compiled
// separately for both, so the check is resolved at compile time.
func isFloat32[T FloatType]() bool {
	var zero T

	return unsafe.Sizeof(zero) == 4
}

func relu[T FloatType](x T) T {
	if x < 0 {
		return 0
	}

	return x
}
