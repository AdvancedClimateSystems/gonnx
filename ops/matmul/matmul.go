package matmul

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var matmul1TypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var matmulTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

// MatMul represents the ONNX matmul operator.
type MatMul struct {
	ops.BaseOperator
}

// newMatMul returns a new MatMul operator.
func newMatMul(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &MatMul{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"matmul",
		),
	}
}

// Init initializes the matmul operator.
func (m *MatMul) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the matmul operator.
func (m *MatMul) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	A := inputs[0]
	B := inputs[1]

	// If both are normal matrices, apply normal matrix multiplication.
	if len(A.Shape()) == 2 && len(B.Shape()) == 2 {
		out, err := tensor.MatMul(A, B)
		if err != nil {
			return nil, err
		}

		return []tensor.Tensor{out}, err
	}

	// If A is a vector, promote to a matrix for the calculation.
	prependedDimension := false
	if len(A.Shape()) == 1 {
		prependedDimension = true

		reshapedA, ok := A.Clone().(tensor.Tensor)
		if !ok {
			return nil, ops.ErrTypeAssert("tensor.Tensor", A.Clone())
		}

		if err := reshapedA.Reshape(1, reshapedA.Shape()[0]); err != nil {
			return nil, err
		}

		A = reshapedA
	}

	// If B is a vector, promote to a matrix for the calculation.
	appendedDimension := false
	if len(B.Shape()) == 1 {
		appendedDimension = true

		reshapedB, ok := B.Clone().(tensor.Tensor)
		if !ok {
			return nil, ops.ErrTypeAssert("tensor.Tensor", B.Clone())
		}

		if err := reshapedB.Reshape(reshapedB.Shape()[0], 1); err != nil {
			return nil, err
		}

		B = reshapedB
	}

	// Now we have to perform batch matrix multiplication. First we need to broadcast
	// the tensor matrices, then we perform matrix multiplication many times.
	A, B, err := m.broadcastTensors(A, B)
	if err != nil {
		return nil, err
	}

	// Perform the batched matrix multiplication on the (possibly broadcasted) tensors.
	out, err := m.batchedMatMul(A, B)
	if err != nil {
		return nil, err
	}

	if prependedDimension {
		currentShape := out.Shape().Clone()
		newShape := currentShape[:len(currentShape)-2]
		newShape = append(newShape, currentShape[len(currentShape)-1])

		if err := out.Reshape(newShape...); err != nil {
			return nil, err
		}
	}

	if appendedDimension {
		currentShape := out.Shape().Clone()
		newShape := currentShape[:len(currentShape)-1]

		if err = out.Reshape(newShape...); err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, err
}

// broadcastTensors broadcasts both tensors for the matmul operator. It is almost identical
// to multidirectional broadcast, but here we need to treat the 2 trailing dimensions as
// matrices, and we do not want to broadcast those. All leading dimensions to the matrices
// are broadcasted the normal way.
func (m *MatMul) broadcastTensors(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
	A, B, err := ops.ReshapeTensorsForMultidirBroadcast(A, B)
	if err != nil {
		return nil, nil, err
	}

	// The trailing 2 dimensions are guaranteed to be matrix dimensions, hence we do not
	// want to broadcast those. All leading dimensions we do want to broadcast.
	shapeA := A.Shape()
	shapeB := B.Shape()

	nMatrixDims := 3

	for axis := len(shapeA) - nMatrixDims; axis >= 0; axis-- {
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
				return nil, nil, ops.ErrIncompatibleDimensions()
			}
		}
	}

	return A, B, nil
}

// batchedMatMul performs the matmul operator on all matrices present in the A and B tensors.
// The trailing two dimensions of the tensors are the matrices that need to be multiplied.
// It is assumed that the tensors are broadcasted accordingly in advance.
func (m *MatMul) batchedMatMul(A, B tensor.Tensor) (tensor.Tensor, error) {
	shapeA := A.Shape()
	shapeB := B.Shape()

	outerShape := append([]int{}, shapeA[:len(shapeA)-2]...)

	// This will be the shape of the output tensor.
	outShape := append([]int{}, outerShape...)
	outShape = append(outShape, shapeA[len(shapeA)-2], shapeB[len(shapeB)-1])
	out := tensor.New(tensor.WithShape(outShape...), tensor.Of(A.Dtype()))

	// Create slices to extract the matrices from the tensors.
	slices := make([]tensor.Slice, len(outerShape))
	for i := 0; i < len(outerShape); i++ {
		slices[i] = ops.NewSlicer(0)
	}

	var err error

	var matrixA, matrixB, matrixOut tensor.Tensor

	for {
		matrixA, err = A.Slice(slices...)
		if err != nil {
			return nil, err
		}

		matrixB, err = B.Slice(slices...)
		if err != nil {
			return nil, err
		}

		matrixOut, err = out.Slice(slices...)
		if err != nil {
			return nil, err
		}

		_, err = tensor.MatMul(matrixA, matrixB, tensor.WithReuse(matrixOut))
		if err != nil {
			return nil, err
		}

		incrementSucceeded := incrementSlices(slices, outerShape)
		if !incrementSucceeded {
			break
		}
	}

	return out, nil
}

// incrementSlices increments all slice by 1. It is used to extract the next matrices
// in the batchedMatMul operation. If the incrementing fails, false is returned.
func incrementSlices(slices []tensor.Slice, shape []int) bool {
	for i := len(shape) - 1; i >= 0; i-- {
		dimSliceStart := slices[i].Start()
		dimSize := shape[i]

		if dimSize == (dimSliceStart + 1) {
			// If we are at the first dimension, we cannot increment the slices anymore.
			if i == 0 {
				return false
			}

			slices[i] = ops.NewSlicer(0) // Else we start again for this dimension.
		} else {
			slices[i] = ops.NewSlicer(dimSliceStart + 1)

			return true
		}
	}

	return false
}
