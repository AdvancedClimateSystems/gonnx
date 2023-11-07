package ops

import (
	"gorgonia.org/tensor"
)

// MultidirectionalBroadcast broadcasts two tensors for a binary operator according to
// the ONNX standards.
func MultidirectionalBroadcast(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	newA, newB, err := ReshapeTensorsForMultidirBroadcast(A, B)
	if err != nil {
		return nil, nil, ErrMultidirBroadcast(A.Shape(), B.Shape(), err)
	}

	newA, newB, err = repeatTensorsForMutltidirBroadcast(newA, newB)
	if err != nil {
		return nil, nil, ErrMultidirBroadcast(A.Shape(), B.Shape(), err)
	}

	return newA, newB, nil
}

// ReshapeTensorsForMultidirBroadcast reshapes the 2 tensors such that they have the same
// number of dimensions. This means that when the number of dimensions do not
// correspond, the shape of the tensor with the smaller number of dimensions gets
// padded with 1's such that it matches the number of dimensions of the other tensor.
// One of the tensors (or both) will always remain the same.
// Example: shapeA=(3, 4) and shapeB=(1, 3, 5, 6) yields shapeNewA=(1, 1, 3, 4).
func ReshapeTensorsForMultidirBroadcast(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	nDimsA := len(A.Shape())
	nDimsB := len(B.Shape())

	switch {
	case nDimsA > nDimsB:
		newB, err := addExtraDimsToTensor(B, nDimsA-nDimsB)
		if err != nil {
			return nil, nil, err
		}

		return A, newB, nil
	case nDimsB > nDimsA:
		newA, err := addExtraDimsToTensor(A, nDimsB-nDimsA)
		if err != nil {
			return nil, nil, err
		}

		return newA, B, nil
	default:
		return A, B, nil
	}
}

// repeatTensorsForMutltidirBroadcast checks along every dimension of both tensors if they have
// the same size. If this is not the case, the dimension that has size 1 is repeated to match
// the dimension of the other. If both sizes are not 1, the tensors cannot be broadcasted to
// each other. It is assumed that both tensors are reshaped accordingly first.
// Example:
//
//	shapeA=(1, 3, 4) and shapeB=(2, 3, 1) yields shapeNewA=(2, 3, 4) and shapeNewB=(2, 3, 4).
func repeatTensorsForMutltidirBroadcast(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	var err error

	shapeA := A.Shape()
	shapeB := B.Shape()
	nDims := len(shapeA)

	for axis := nDims - 1; axis >= 0; axis-- {
		sizeDimA := shapeA[axis]
		sizeDimB := shapeB[axis]

		if sizeDimA != sizeDimB {
			switch {
			case sizeDimA == 1:
				A, err = tensor.Repeat(A, axis, sizeDimB)
				if err != nil {
					return nil, nil, err
				}

			case sizeDimB == 1:
				B, err = tensor.Repeat(B, axis, sizeDimA)
				if err != nil {
					return nil, nil, err
				}

			default:
				return nil, nil, ErrIncompatibleDimensions()
			}
		}
	}

	return A, B, nil
}

// addExtraDimsToTensor adds a given number of dimensions to the shape of a tensor (in the front).
// All extra dimensions are given size one (otherwise the tensor cannot be reshaped).
// The given tensor is cloned such that the tensor is not modified in place.
// Example: if we add 2 extra dimensions to shape (2, 3) we get shape (1, 1, 2, 3).
func addExtraDimsToTensor(t tensor.Tensor, nExtraDims int) (tensor.Tensor, error) {
	t, ok := t.Clone().(tensor.Tensor)
	if !ok {
		return nil, ErrTypeAssert("tensor.Tensor", t.Clone())
	}

	newShape := []int{}
	for i := 0; i < nExtraDims; i++ {
		newShape = append(newShape, 1)
	}

	newShape = append(newShape, t.Shape()...)

	if err := t.Reshape(newShape...); err != nil {
		return nil, err
	}

	return t, nil
}
