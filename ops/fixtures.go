package ops

import (
	"gorgonia.org/tensor"
)

// InputFixture is a function that generates inputs for ops. Useful in testing
type InputFixture func() []tensor.Tensor

// Float32TensorFixture returns a float32 backed gorgonia node. It initializes all its values
// using tensor.Range.
func Float32TensorFixture(shp ...int) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(shp...),
		tensor.WithBacking(tensor.Range(tensor.Float32, 0, NElements(shp...))),
	)
}

// TensorWithBackingFixture returns a gorgonia node with a tensor using the given backing.
func TensorWithBackingFixture(b interface{}, shp ...int) tensor.Tensor {
	return tensor.New(tensor.WithShape(shp...), tensor.WithBacking(b))
}

// TensorInputsFixture returns a list with a given number of tensors.
func TensorInputsFixture(nTensors int) []tensor.Tensor {
	result := make([]tensor.Tensor, nTensors)
	for i := 0; i < nTensors; i++ {
		result[i] = tensor.New(tensor.WithShape(1), tensor.WithBacking([]float32{0.0}))
	}

	return result
}
