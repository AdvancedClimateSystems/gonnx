package matmul

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinMatMul1Inputs = 2
	MaxMatMul1Inputs = 2
)

// MatMul1 represents the ONNX matmul operator.
type MatMul1 struct{}

// newMatMul1 returns a new MatMul1 operator.
func newMatMul1() ops.Operator {
	return &MatMul1{}
}

// Init initializes the matmul operator.
func (m *MatMul1) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the matmul operator.
func (m *MatMul1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (m *MatMul1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(m, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (m *MatMul1) GetMinInputs() int {
	return MinMatMul1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (m *MatMul1) GetMaxInputs() int {
	return MaxMatMul1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (m *MatMul1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (m *MatMul1) String() string {
	return "matmul13 operator"
}

// broadcastTensors broadcasts both tensors for the matmul operator. It is almost identical
// to multidirectional broadcast, but here we need to treat the 2 trailing dimensions as
// matrices, and we do not want to broadcast those. All leading dimensions to the matrices
// are broadcasted the normal way.
func (m *MatMul1) broadcastTensors(A, B tensor.Tensor) (tensor.Tensor, tensor.Tensor, error) {
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
func (m *MatMul1) batchedMatMul(A, B tensor.Tensor) (tensor.Tensor, error) {
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
