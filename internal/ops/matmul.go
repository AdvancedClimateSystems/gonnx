package ops

import (
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// MatMulInto computes out = op(a)·op(b) for row-major matrices, where op(a) is an m×k
// matrix and op(b) a k×n matrix. When transA is set, a holds the k×m matrix that op
// transposes; likewise for transB, where b holds an n×k matrix.
//
// Every output element is summed over k in ascending order, so results do not depend on
// how the loops are blocked.
func MatMulInto[T tensor.Numeric](out, a, b []T, m, n, k int, transA, transB bool) {
	out = out[:m*n]

	if transB {
		matMulTransposedB(out, a, b, m, n, k, transA)

		return
	}

	matMulRows(out, a, b, m, n, k, transA)
}

// matMulRows computes each output row as a sum of scaled rows of b. The innermost loop
// runs over contiguous output elements, four rows of b at a time.
func matMulRows[T tensor.Numeric](out, a, b []T, m, n, k int, transA bool) {
	elementA := func(i, l int) T { return a[i*k+l] }
	if transA {
		elementA = func(i, l int) T { return a[l*m+i] }
	}

	for i := 0; i < m; i++ {
		row := out[i*n : i*n+n]
		for j := range row {
			row[j] = 0
		}

		l := 0
		for ; l+4 <= k; l += 4 {
			a0, a1, a2, a3 := elementA(i, l), elementA(i, l+1), elementA(i, l+2), elementA(i, l+3)
			b0 := b[l*n:][:len(row)]
			b1 := b[(l+1)*n:][:len(row)]
			b2 := b[(l+2)*n:][:len(row)]
			b3 := b[(l+3)*n:][:len(row)]

			// Two output elements per iteration, with loads and products issued before the
			// additions, for the same reason as in matMulTransposedB.
			j := 0
			for ; j+2 <= len(row); j += 2 {
				sum0, sum1 := row[j], row[j+1]
				x00, x10, x20, x30 := b0[j], b1[j], b2[j], b3[j]
				x01, x11, x21, x31 := b0[j+1], b1[j+1], b2[j+1], b3[j+1]
				p00, p10, p20, p30 := a0*x00, a1*x10, a2*x20, a3*x30
				p01, p11, p21, p31 := a0*x01, a1*x11, a2*x21, a3*x31
				sum0 += p00
				sum1 += p01
				sum0 += p10
				sum1 += p11
				sum0 += p20
				sum1 += p21
				sum0 += p30
				sum1 += p31
				row[j], row[j+1] = sum0, sum1
			}

			for ; j < len(row); j++ {
				sum := row[j]
				sum += a0 * b0[j]
				sum += a1 * b1[j]
				sum += a2 * b2[j]
				sum += a3 * b3[j]
				row[j] = sum
			}
		}

		for ; l < k; l++ {
			value := elementA(i, l)
			bRow := b[l*n:][:len(row)]

			for j := range row {
				row[j] += value * bRow[j]
			}
		}
	}
}

// matMulTransposedB computes each output element as the dot product of a row of op(a) and
// a row of b, four output elements at a time so the four sums are independent.
func matMulTransposedB[T tensor.Numeric](out, a, b []T, m, n, k int, transA bool) {
	var column []T
	if transA {
		column = make([]T, k)
	}

	for i := 0; i < m; i++ {
		var rowA []T

		if transA {
			for l := range column {
				column[l] = a[l*m+i]
			}

			rowA = column
		} else {
			rowA = a[i*k : i*k+k]
		}

		row := out[i*n : i*n+n]

		j := 0
		for ; j+4 <= n; j += 4 {
			// Slicing every row to the length of rowA lets the compiler drop the bounds
			// checks in the loop below.
			b0 := b[j*k:][:len(rowA)]
			b1 := b[(j+1)*k:][:len(rowA)]
			b2 := b[(j+2)*k:][:len(rowA)]
			b3 := b[(j+3)*k:][:len(rowA)]

			var sum0, sum1, sum2, sum3 T

			// Two values of l per iteration, with all loads and products issued before
			// the additions: in-order cores stall when a result is used right after the
			// instruction that produces it.
			l := 0
			for ; l+2 <= len(rowA); l += 2 {
				v0, v1 := rowA[l], rowA[l+1]
				x00, x10, x20, x30 := b0[l], b1[l], b2[l], b3[l]
				x01, x11, x21, x31 := b0[l+1], b1[l+1], b2[l+1], b3[l+1]
				p00, p10, p20, p30 := v0*x00, v0*x10, v0*x20, v0*x30
				p01, p11, p21, p31 := v1*x01, v1*x11, v1*x21, v1*x31
				sum0 += p00
				sum1 += p10
				sum2 += p20
				sum3 += p30
				sum0 += p01
				sum1 += p11
				sum2 += p21
				sum3 += p31
			}

			for ; l < len(rowA); l++ {
				value := rowA[l]
				sum0 += value * b0[l]
				sum1 += value * b1[l]
				sum2 += value * b2[l]
				sum3 += value * b3[l]
			}

			row[j], row[j+1], row[j+2], row[j+3] = sum0, sum1, sum2, sum3
		}

		for ; j < n; j++ {
			bRow := b[j*k:][:len(rowA)]

			var sum T

			for l, value := range rowA {
				sum += value * bRow[l]
			}

			row[j] = sum
		}
	}
}

// MatMul multiplies two tensors following numpy's matmul semantics: 1-D inputs are
// treated as a row (A) or column (B) vector, and the dimensions in front of the last two
// are broadcast.
func MatMul(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if A.DataType() != B.DataType() {
		return nil, ErrDataTypeMismatch(A.DataType(), B.DataType())
	}

	switch A.DataType() {
	case tensor.Float32:
		return matMul[float32](A, B)
	case tensor.Float64:
		return matMul[float64](A, B)
	case tensor.Int32:
		return matMul[int32](A, B)
	case tensor.Int64:
		return matMul[int64](A, B)
	case tensor.Uint32:
		return matMul[uint32](A, B)
	case tensor.Uint64:
		return matMul[uint64](A, B)
	default:
		return nil, ErrInvalidType(A.DataType())
	}
}

func matMul[T tensor.Numeric](A, B *tensor.Tensor) (*tensor.Tensor, error) {
	shapeA, shapeB := A.Shape(), B.Shape()
	if len(shapeA) == 0 || len(shapeB) == 0 {
		return nil, ErrIncompatibleDimensions()
	}

	vectorA, vectorB := len(shapeA) == 1, len(shapeB) == 1

	if vectorA {
		shapeA = tensor.Shape{1, shapeA[0]}
	}

	if vectorB {
		shapeB = tensor.Shape{shapeB[0], 1}
	}

	m, k := shapeA[len(shapeA)-2], shapeA[len(shapeA)-1]
	n := shapeB[len(shapeB)-1]

	if shapeB[len(shapeB)-2] != k {
		return nil, ErrIncompatibleDimensions()
	}

	batchA, batchB := shapeA[:len(shapeA)-2], shapeB[:len(shapeB)-2]

	batchShape, err := MultidirectionalBroadcastShape(batchA, batchB)
	if err != nil {
		return nil, err
	}

	nBatches := batchShape.Size()
	matricesA := batchOffsets(batchA, batchShape)
	matricesB := batchOffsets(batchB, batchShape)

	a, b := tensor.Data[T](A), tensor.Data[T](B)
	out := make([]T, nBatches*m*n)

	for batch := 0; batch < nBatches; batch++ {
		offsetA := matricesA[batch] * m * k
		offsetB := matricesB[batch] * k * n
		MatMulInto(out[batch*m*n:], a[offsetA:offsetA+m*k], b[offsetB:offsetB+k*n], m, n, k, false, false)
	}

	outShape := append(batchShape.Clone(), m, n)

	switch {
	case vectorA && vectorB:
		outShape = outShape[:len(outShape)-2]
	case vectorA:
		outShape = append(outShape[:len(outShape)-2], n)
	case vectorB:
		outShape = outShape[:len(outShape)-1]
	}

	return tensor.New(out, outShape...), nil
}

// batchOffsets returns, for every position in the broadcast batch shape, which matrix of
// a tensor with the given batch dimensions is used there.
func batchOffsets(batch, batchShape tensor.Shape) []int {
	indices := make([]int, batch.Size())
	for i := range indices {
		indices[i] = i
	}

	offsets := make([]int, batchShape.Size())
	stridedCopy(offsets, indices, batchShape, broadcastStrides(batch, batchShape), 0)

	return offsets
}

// Gemm computes alpha·op(A)·op(B) + beta·C, where op transposes its matrix when the
// corresponding flag is set. C is optional (nil) and is broadcast unidirectionally to the
// shape of the product.
func Gemm(A, B, C *tensor.Tensor, alpha, beta float32, transA, transB bool) (*tensor.Tensor, error) {
	if A.DataType() != B.DataType() {
		return nil, ErrDataTypeMismatch(A.DataType(), B.DataType())
	}

	if C != nil && C.DataType() != A.DataType() {
		return nil, ErrDataTypeMismatch(A.DataType(), C.DataType())
	}

	switch A.DataType() {
	case tensor.Float32:
		return gemm[float32](A, B, C, alpha, beta, transA, transB)
	case tensor.Float64:
		return gemm[float64](A, B, C, alpha, beta, transA, transB)
	case tensor.Int32:
		return gemm[int32](A, B, C, alpha, beta, transA, transB)
	case tensor.Int64:
		return gemm[int64](A, B, C, alpha, beta, transA, transB)
	case tensor.Uint32:
		return gemm[uint32](A, B, C, alpha, beta, transA, transB)
	case tensor.Uint64:
		return gemm[uint64](A, B, C, alpha, beta, transA, transB)
	default:
		return nil, ErrInvalidType(A.DataType())
	}
}

func gemm[T tensor.Numeric](A, B, C *tensor.Tensor, alpha, beta float32, transA, transB bool) (*tensor.Tensor, error) {
	if A.Rank() != 2 || B.Rank() != 2 {
		return nil, ErrIncompatibleDimensions()
	}

	m, k := A.Shape()[0], A.Shape()[1]
	if transA {
		m, k = k, m
	}

	kB, n := B.Shape()[0], B.Shape()[1]
	if transB {
		kB, n = n, kB
	}

	if k != kB {
		return nil, ErrIncompatibleDimensions()
	}

	out := make([]T, m*n)
	MatMulInto(out, tensor.Data[T](A), tensor.Data[T](B), m, n, k, transA, transB)

	return finishGemm(out, m, n, C, T(alpha), T(beta))
}

// GemmPacked computes alpha·A·Bᵀ + beta·C like Gemm, for a float32 matrix A and packed
// weights B.
func GemmPacked(A *tensor.Tensor, B *PackedMatrix, C *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error) {
	if A.DataType() != tensor.Float32 {
		return nil, ErrDataTypeMismatch(A.DataType(), tensor.Float32)
	}

	if C != nil && C.DataType() != tensor.Float32 {
		return nil, ErrDataTypeMismatch(tensor.Float32, C.DataType())
	}

	n, k := B.Dims()
	if A.Rank() != 2 || A.Shape()[1] != k {
		return nil, ErrIncompatibleDimensions()
	}

	m := A.Shape()[0]
	out := make([]float32, m*n)
	B.MatMul(out, tensor.Data[float32](A), m)

	return finishGemm(out, m, n, C, alpha, beta)
}

// finishGemm scales the m×n product out by alpha and adds beta·C, with C broadcast to
// the shape of the product.
func finishGemm[T tensor.Numeric](out []T, m, n int, C *tensor.Tensor, alpha, beta T) (*tensor.Tensor, error) {
	if alpha != 1 {
		for i := range out {
			out[i] *= alpha
		}
	}

	shape := tensor.Shape{m, n}

	if C != nil {
		if _, err := UnidirectionalBroadcastShape(shape, C.Shape()); err != nil {
			return nil, err
		}

		addScaled(out, shape, tensor.Data[T](C), C.Shape(), beta)
	}

	return tensor.New(out, shape...), nil
}

// MatMulPacked multiplies A with packed weights B following numpy's matmul semantics, for
// a float32 tensor A and a 2D matrix B: the last dimension of A is multiplied with B.
func MatMulPacked(A *tensor.Tensor, B *PackedMatrix) (*tensor.Tensor, error) {
	if A.DataType() != tensor.Float32 {
		return nil, ErrDataTypeMismatch(A.DataType(), tensor.Float32)
	}

	n, k := B.Dims()
	if A.Rank() == 0 || A.Shape()[A.Rank()-1] != k {
		return nil, ErrIncompatibleDimensions()
	}

	rows := A.Size() / k
	out := make([]float32, rows*n)
	B.MatMul(out, tensor.Data[float32](A), rows)

	shape := append(A.Shape()[:A.Rank()-1].Clone(), n)

	return tensor.New(out, shape...), nil
}

// TransposedProduct returns a function computing out = a·Bᵀ for an m×k matrix a, with
// B the n×k matrix weights. It uses packed, the packed form of weights, when it is not
// nil and T is float32.
func TransposedProduct[T FloatType](weights []T, n, k int, packed *PackedMatrix) func(out, a []T, m int) {
	if packed != nil {
		var float32Product any = func(out, a []float32, m int) { packed.MatMul(out, a, m) }
		if product, ok := float32Product.(func(out, a []T, m int)); ok {
			return product
		}
	}

	return func(out, a []T, m int) { MatMulInto(out, a, weights, m, n, k, false, true) }
}

// addScaled adds scale·c to the m×n matrix out, where c is broadcast to the shape of out.
func addScaled[T tensor.Numeric](out []T, shape tensor.Shape, c []T, cShape tensor.Shape, scale T) {
	strides := broadcastStrides(cShape, shape)
	m, n := shape[0], shape[1]

	for i := 0; i < m; i++ {
		row := out[i*n : i*n+n]
		position := i * strides[0]

		if scale == 1 {
			for j := range row {
				row[j] += c[position]
				position += strides[1]
			}

			continue
		}

		for j := range row {
			row[j] += c[position] * scale
			position += strides[1]
		}
	}
}
