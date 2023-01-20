package ops

import (
	"fmt"

	"gorgonia.org/tensor"
)

// UnidirectionalBroadcast tries to broadcast tensor B to tensor A according to the ONNX standards.
func UnidirectionalBroadcast(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {

	reshapedB, err := reshapeTensorsForUnidirBroadcast(A, B)
	if err != nil {
		return nil, nil, fmt.Errorf(UnidirBroadcastErrTemplate, A.Shape(), B.Shape())
	}

	newB, err := repeatTensorsForUnidirBroadcast(A, reshapedB)
	if err != nil {
		return nil, nil, fmt.Errorf(UnidirBroadcastErrTemplate, A.Shape(), B.Shape())
	}

	return A, newB, nil
}

// reshapeTensorsForUnidirBroadcast reshapes the B tensor to match the number of dimensions
// of the A tensor. New dimensions of size 1 are added to the front.
// Example: shapeA=(2, 3, 4) and shapeB=(3, 4) yields shapeNewB=(1, 3, 4).
func reshapeTensorsForUnidirBroadcast(A, B tensor.Tensor) (tensor.Tensor, error) {
	nDimsA := len(A.Shape())
	nDimsB := len(B.Shape())

	switch {
	case nDimsA > nDimsB:
		return addExtraDimsToTensor(B, nDimsA-nDimsB)
	case nDimsA == nDimsB:
		return B, nil
	default:
		return nil, fmt.Errorf("tensor B may not have more dimensions than tensor A")
	}
}

// repeatTensorsForUnidirBroadcast broadcasts tensor B such that it corresponds with the
// shape of tensor A. Assumes the B tensor has already been reshaped such that it has
// the same number of dimensions as tensor A.
// Example: shapeA=(2, 3, 4) and shapeB=(1, 3, 4) yields shapeNewB=(2, 3, 4).
func repeatTensorsForUnidirBroadcast(A, B tensor.Tensor) (tensor.Tensor, error) {
	var err error
	shapeA := A.Shape()
	shapeB := B.Shape()

	// Repeatedly repeat the B tensor along dimensions of size 1.
	for axis := len(shapeA) - 1; axis >= 0; axis-- {
		sizeDimA := shapeA[axis]
		sizeDimB := shapeB[axis]

		if sizeDimA != sizeDimB {
			if sizeDimB != 1 {
				return nil, fmt.Errorf("incompatible dimensions")
			}

			B, err = tensor.Repeat(B, axis, sizeDimA)
			if err != nil {
				return nil, err
			}
		}
	}

	return B, nil
}
