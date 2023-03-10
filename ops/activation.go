package ops

import "gorgonia.org/tensor"

// Activation is an activation function.
type Activation func(n tensor.Tensor) (tensor.Tensor, error)

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
