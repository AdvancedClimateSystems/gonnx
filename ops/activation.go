package ops

import (
	"gorgonia.org/tensor"
)

// Activation is an activation function.
type Activation func(n tensor.Tensor) (tensor.Tensor, error)

// activations maps strings to the activation function. This is
// used by operators like LSTM, GRU and RNN.
var activations = map[string]Activation{
	"tanh":    Tanh,
	"sigmoid": Sigmoid,
	"relu":    ReLU,
}

func GetActivation(activation string) (Activation, error) {
	if a, ok := activations[activation]; ok {
		return a, nil
	}

	return nil, ErrActivationNotImplemented(activation)
}

// Tanh performs the tanh operation on a tensor.
func Tanh(X tensor.Tensor) (tensor.Tensor, error) {
	return tensor.Tanh(X)
}

// Sigmoid performs the sigmoid operation on a tensor.
func Sigmoid(X tensor.Tensor) (tensor.Tensor, error) {
	negX, err := tensor.Neg(X)
	if err != nil {
		return nil, err
	}

	expX, err := tensor.Exp(negX)
	if err != nil {
		return nil, err
	}

	typedOne, err := GetValueAsTensorType(1.0, expX.Dtype())
	if err != nil {
		return nil, err
	}

	numeratorX, err := tensor.Add(typedOne, expX)
	if err != nil {
		return nil, err
	}

	return tensor.Div(typedOne, numeratorX)
}

// ReLU performs the ReLU operation on a tensor.
func ReLU(X tensor.Tensor) (tensor.Tensor, error) {
	typedZero, err := GetValueAsTensorType(0.0, X.Dtype())
	if err != nil {
		return nil, err
	}

	comparison, err := tensor.Gt(X, typedZero, tensor.AsSameType())
	if err != nil {
		return nil, err
	}

	return tensor.Mul(X, comparison)
}
