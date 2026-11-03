package ops

import (
	"math/rand"
	"testing"

	"github.com/advancedclimatesystems/gonnx/tensor"
	"github.com/stretchr/testify/assert"
)

// naiveMatMul multiplies op(a) (m×k) with op(b) (k×n), summing over k in ascending order.
func naiveMatMul(a, b []float32, m, n, k int, transA, transB bool) []float32 {
	out := make([]float32, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32

			for l := 0; l < k; l++ {
				valueA := a[i*k+l]
				if transA {
					valueA = a[l*m+i]
				}

				valueB := b[l*n+j]
				if transB {
					valueB = b[j*k+l]
				}

				sum += valueA * valueB
			}

			out[i*n+j] = sum
		}
	}

	return out
}

func randomFloat32s(r *rand.Rand, n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = r.Float32()*2 - 1
	}

	return values
}

func TestMatMulIntoMatchesNaive(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	for _, m := range []int{1, 2, 5} {
		for _, n := range []int{1, 3, 4, 9} {
			for _, k := range []int{1, 4, 7} {
				for _, transA := range []bool{false, true} {
					for _, transB := range []bool{false, true} {
						a := randomFloat32s(r, m*k)
						b := randomFloat32s(r, k*n)
						out := make([]float32, m*n)

						MatMulInto(out, a, b, m, n, k, transA, transB)

						assert.Equal(t, naiveMatMul(a, b, m, n, k, transA, transB), out, "m=%d n=%d k=%d transA=%v transB=%v", m, n, k, transA, transB)
					}
				}
			}
		}
	}
}

func TestMatMul(t *testing.T) {
	tests := []struct {
		shapeA        []int
		shapeB        []int
		expectedShape tensor.Shape
		expected      []float32
	}{
		{[]int{3}, []int{3}, tensor.Shape{}, []float32{5}},
		{[]int{2}, []int{2, 3}, tensor.Shape{3}, []float32{3, 4, 5}},
		{[]int{2, 3}, []int{3}, tensor.Shape{2}, []float32{5, 14}},
		{[]int{2, 3}, []int{3, 2}, tensor.Shape{2, 2}, []float32{10, 13, 28, 40}},
		{
			[]int{2, 1, 1, 2}, []int{3, 2, 1},
			tensor.Shape{2, 3, 1, 1},
			[]float32{1, 3, 5, 3, 13, 23},
		},
	}

	for _, test := range tests {
		A := Float32TensorFixture(test.shapeA...)
		B := Float32TensorFixture(test.shapeB...)

		out, err := MatMul(A, B)

		assert.Nil(t, err)
		assert.Equal(t, test.expectedShape, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}
}

func TestMatMulErrors(t *testing.T) {
	_, err := MatMul(Float32TensorFixture(2, 3), Float32TensorFixture(2, 3))
	assert.Equal(t, ErrIncompatibleDimensions(), err)

	_, err = MatMul(Float32TensorFixture(2, 3), tensor.New([]float64{1, 2, 3}, 3))
	assert.ErrorIs(t, err, ErrUnsupportedType)
}

func TestGemm(t *testing.T) {
	A := tensor.New([]float32{1, 2, 3, 4}, 2, 2)
	B := tensor.New([]float32{1, 0, 0, 1}, 2, 2)

	tests := []struct {
		C        *tensor.Tensor
		alpha    float32
		beta     float32
		transA   bool
		expected []float32
	}{
		{nil, 1, 1, false, []float32{1, 2, 3, 4}},
		{nil, 2, 1, true, []float32{2, 6, 4, 8}},
		{tensor.New([]float32{10, 20}, 2), 1, 1, false, []float32{11, 22, 13, 24}},
		{tensor.New([]float32{10, 20}, 2, 1), 1, 0.5, false, []float32{6, 7, 13, 14}},
		{tensor.Scalar(float32(1)), 1, 1, false, []float32{2, 3, 4, 5}},
	}

	for _, test := range tests {
		out, err := Gemm(A, B, test.C, test.alpha, test.beta, test.transA, false)

		assert.Nil(t, err)
		assert.Equal(t, tensor.Shape{2, 2}, out.Shape())
		assert.Equal(t, test.expected, out.Data())
	}
}

func BenchmarkMatMulInto(b *testing.B) {
	benchmarks := []struct {
		name    string
		m, n, k int
		transB  bool
	}{
		{"recurrent_1x256x768_transB", 1, 768, 256, true},
		{"projection_18x5x768_transB", 18, 768, 5, true},
		{"rows_1x100x300", 1, 300, 100, false},
	}

	r := rand.New(rand.NewSource(1))

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			a := randomFloat32s(r, bm.m*bm.k)
			weights := randomFloat32s(r, bm.k*bm.n)
			out := make([]float32, bm.m*bm.n)

			b.SetBytes(int64(4 * len(weights)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				MatMulInto(out, a, weights, bm.m, bm.n, bm.k, false, bm.transB)
			}
		})
	}
}
