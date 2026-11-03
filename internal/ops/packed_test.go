package ops

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPackedMatMulMatchesMatMulInto(t *testing.T) {
	if !hasSIMD {
		t.Skip("no SIMD kernel on this platform")
	}

	r := rand.New(rand.NewSource(1))

	for _, m := range []int{1, 3} {
		for _, n := range []int{1, 15, 16, 17, 40, 768} {
			for _, k := range []int{1, 3, 4, 5, 8, 256, 259} {
				a := randomFloat32s(r, m*k)
				b := randomFloat32s(r, n*k)

				expected := make([]float32, m*n)
				MatMulInto(expected, a, b, m, n, k, false, true)

				out := make([]float32, m*n)
				PackTransposed(b, n, k).MatMul(out, a, m)
				assert.Equal(t, expected, out, "transposed m=%d n=%d k=%d", m, n, k)

				// The same values as a k×n matrix.
				bT := make([]float32, k*n)

				for j := 0; j < n; j++ {
					for l := 0; l < k; l++ {
						bT[l*n+j] = b[j*k+l]
					}
				}

				out = make([]float32, m*n)
				Pack(bT, k, n).MatMul(out, a, m)
				assert.Equal(t, expected, out, "untransposed m=%d n=%d k=%d", m, n, k)
			}
		}
	}
}

func TestPackWithoutSIMD(t *testing.T) {
	original := hasSIMD
	hasSIMD = false

	defer func() { hasSIMD = original }()

	assert.Nil(t, PackTransposed([]float32{1, 2}, 1, 2))
}

func BenchmarkPackedMatMul(b *testing.B) {
	if !hasSIMD {
		b.Skip("no SIMD kernel on this platform")
	}

	benchmarks := []struct {
		name string
		n, k int
	}{
		{"cache_resident_64x64", 64, 64},
		{"recurrent_768x256", 768, 256},
	}

	r := rand.New(rand.NewSource(1))

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			a := randomFloat32s(r, bm.k)
			weights := randomFloat32s(r, bm.n*bm.k)
			packed := PackTransposed(weights, bm.n, bm.k)
			out := make([]float32, bm.n)

			b.SetBytes(int64(4 * len(weights)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				packed.MatMul(out, a, 1)
			}
		})
	}
}
