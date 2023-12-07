package ops

import (
	"gorgonia.org/tensor"
)

// SequenceProcessDirection is the direction in which a sequential input is processed.
// We can process sequential inputs forward (from first to last), in reverse (from
// last to first) or bidirectional (which is both forward and reverse added together).
type SequenceProcessDirection string

const (
	Forward       SequenceProcessDirection = "forward"
	Reverse       SequenceProcessDirection = "reverse"
	Bidirectional SequenceProcessDirection = "bidirectional"
)

// These constants define attributes that are applicable to GRU, LSTM and RNN operators.
const (
	ActivationAlphaAttr = "activation_alpha"
	ActivationBetaAttr  = "activation_beta"
	ActivationsAttr     = "activations"
	ClipAttr            = "clip"
	DirectionAttr       = "direction"
	HiddenSizeAttr      = "hidden_size"
)

// extractMatrices extracts a given number of matrices from tensor M.
// M contains concatenated matrices along a certain dimension.
// M is assumed to have a shape of (num_directions, nMatrices * hidden_size, ...) and we extract the
// by slicing over the 'nMatrices * hidden_size' dimension.
// This method is specific for recurrent operators RNN, GRU and LSTM.
func ExtractMatrices(M tensor.Tensor, nMatrices, nDimensions, hiddenSize int) ([]tensor.Tensor, error) {
	dirSlice := NewSlicer(0)
	matrices := make([]tensor.Tensor, nMatrices)

	for i := 0; i < nMatrices; i++ {
		hiddenSlice := NewSlicer(i*hiddenSize, (i+1)*hiddenSize)

		allSlices := make([]tensor.Slice, nDimensions)
		allSlices[0] = dirSlice
		allSlices[1] = hiddenSlice

		for i := 2; i < nDimensions; i++ {
			allSlices[i] = nil
		}

		m, err := M.Slice(allSlices...)
		if err != nil {
			return nil, err
		}

		matrices[i] = m
	}

	return matrices, nil
}

// ZeroTensor returns a tensor filled with zeros with the given shape.
func ZeroTensor(shape ...int) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(shape...),
		tensor.WithBacking(Zeros(NElements(shape...))),
	)
}

// OnesTensor returns a new tensor with the same shape as the given tensor intialized with all ones.
func OnesTensor(t tensor.Tensor) tensor.Tensor {
	return tensor.New(
		tensor.WithShape(t.Shape()...),
		tensor.WithBacking(Ones(NElements(t.Shape()...))),
	)
}
