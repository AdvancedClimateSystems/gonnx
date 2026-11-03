package ops

import (
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// BroadcastType describes how the shapes of the inputs of an operator are aligned.
type BroadcastType int

// The broadcast types an operator can use.
const (
	NoBroadcasting               BroadcastType = 0
	UnidirectionalBroadcasting   BroadcastType = 1
	MultidirectionalBroadcasting BroadcastType = 2
)

// MultidirectionalBroadcastShape returns the shape two tensors broadcast to according to
// the ONNX standard: shapes are aligned on their last dimension, and dimensions of size 1
// stretch to the size of the other tensor.
// Example: shapes (1, 4, 5) and (2, 3, 1, 1) broadcast to (2, 3, 4, 5).
func MultidirectionalBroadcastShape(shapeA, shapeB tensor.Shape) (tensor.Shape, error) {
	rank := max(len(shapeA), len(shapeB))
	shape := make(tensor.Shape, rank)

	for axis := range shape {
		sizeA := dimAligned(shapeA, axis, rank)
		sizeB := dimAligned(shapeB, axis, rank)

		switch {
		case sizeA == sizeB, sizeB == 1:
			shape[axis] = sizeA
		case sizeA == 1:
			shape[axis] = sizeB
		default:
			return nil, ErrMultidirBroadcast(shapeA, shapeB, ErrIncompatibleDimensions())
		}
	}

	return shape, nil
}

// UnidirectionalBroadcastShape checks that a tensor with shapeB can be broadcast to
// shapeA, and returns shapeA.
func UnidirectionalBroadcastShape(shapeA, shapeB tensor.Shape) (tensor.Shape, error) {
	if len(shapeB) > len(shapeA) {
		return nil, ErrUnidirBroadcast(shapeA, shapeB)
	}

	for axis := range shapeA {
		sizeB := dimAligned(shapeB, axis, len(shapeA))
		if sizeB != shapeA[axis] && sizeB != 1 {
			return nil, ErrUnidirBroadcast(shapeA, shapeB)
		}
	}

	return shapeA, nil
}

// dimAligned returns the size of the dimension of shape that lines up with axis when the
// shape is right-aligned to a shape of the given rank. Missing leading dimensions are 1.
func dimAligned(shape tensor.Shape, axis, rank int) int {
	i := axis - (rank - len(shape))
	if i < 0 {
		return 1
	}

	return shape[i]
}

// broadcastStrides returns the element strides with which a tensor of the given shape is
// read when it is broadcast to outShape. Broadcast dimensions get stride 0.
func broadcastStrides(shape, outShape tensor.Shape) []int {
	strides := make([]int, len(outShape))
	offset := len(outShape) - len(shape)
	stride := 1

	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] != 1 {
			strides[offset+i] = stride
		}

		stride *= shape[i]
	}

	return strides
}

// BroadcastTo returns a tensor with the given shape holding the elements of t, repeated
// along the dimensions that are broadcast. The shape must be a valid unidirectional
// broadcast target for t.
func BroadcastTo(t *tensor.Tensor, shape tensor.Shape) (*tensor.Tensor, error) {
	if _, err := UnidirectionalBroadcastShape(shape, t.Shape()); err != nil {
		return nil, err
	}

	if t.Shape().Equal(shape) {
		return t, nil
	}

	return Strided(t, shape, broadcastStrides(t.Shape(), shape), 0)
}

// MultidirectionalBroadcast broadcasts two tensors to their common shape according to the
// ONNX standard. Both returned tensors have that shape.
func MultidirectionalBroadcast(A, B *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	shape, err := MultidirectionalBroadcastShape(A.Shape(), B.Shape())
	if err != nil {
		return nil, nil, err
	}

	newA, err := BroadcastTo(A, shape)
	if err != nil {
		return nil, nil, err
	}

	newB, err := BroadcastTo(B, shape)
	if err != nil {
		return nil, nil, err
	}

	return newA, newB, nil
}

// UnidirectionalBroadcast broadcasts tensor B to the shape of tensor A according to the
// ONNX standard. A is returned unchanged.
func UnidirectionalBroadcast(A, B *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	newB, err := BroadcastTo(B, A.Shape())
	if err != nil {
		return nil, nil, ErrUnidirBroadcast(A.Shape(), B.Shape())
	}

	return A, newB, nil
}

// ReshapeTensorsForMultidirBroadcast reshapes the 2 tensors such that they have the same
// number of dimensions, by prepending dimensions of size 1 to the tensor with the lowest
// rank. Example: shapeA=(3, 4) and shapeB=(1, 3, 5, 6) yields shapeNewA=(1, 1, 3, 4).
func ReshapeTensorsForMultidirBroadcast(A, B *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	nDimsA := A.Rank()
	nDimsB := B.Rank()

	switch {
	case nDimsA > nDimsB:
		newB, err := AddExtraDimsToTensor(B, nDimsA-nDimsB)

		return A, newB, err
	case nDimsB > nDimsA:
		newA, err := AddExtraDimsToTensor(A, nDimsB-nDimsA)

		return newA, B, err
	default:
		return A, B, nil
	}
}

// AddExtraDimsToTensor prepends a given number of dimensions of size 1 to the shape of a
// tensor. Example: adding 2 extra dimensions to shape (2, 3) gives shape (1, 1, 2, 3).
func AddExtraDimsToTensor(t *tensor.Tensor, nExtraDims int) (*tensor.Tensor, error) {
	newShape := make([]int, nExtraDims, nExtraDims+t.Rank())
	for i := range newShape {
		newShape[i] = 1
	}

	return t.Reshape(append(newShape, t.Shape()...)...)
}
