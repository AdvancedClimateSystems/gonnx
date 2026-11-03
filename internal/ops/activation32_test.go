package ops

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// ulpDistance returns how many float32 values lie between a and b.
func ulpDistance(a, b float32) int64 {
	orderedBits := func(f float32) int64 {
		bits := int64(math.Float32bits(f))
		if bits&(1<<31) != 0 {
			return -(bits &^ (1 << 31))
		}

		return bits
	}

	distance := orderedBits(a) - orderedBits(b)
	if distance < 0 {
		return -distance
	}

	return distance
}

// maxULPError returns the largest distance between f32 and the float32 rounding of f64
// over evenly spaced arguments in [from, to], ignoring results in the subnormal range
// where float32 has fewer significant bits.
func maxULPError(f32 func(float32) float32, f64 func(float64) float64, from, to float64, n int) int64 {
	var worst int64

	for i := 0; i <= n; i++ {
		x := float32(from + (to-from)*float64(i)/float64(n))

		expected := float32(f64(float64(x)))
		if math.Abs(float64(expected)) < math.SmallestNonzeroFloat32*(1<<23) {
			continue
		}

		worst = max(worst, ulpDistance(f32(x), expected))
	}

	return worst
}

func TestActivation32Accuracy(t *testing.T) {
	sigmoid64 := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

	tests := []struct {
		name     string
		f32      func(float32) float32
		f64      func(float64) float64
		from, to float64
		maxULP   int64
	}{
		{"exp", exp32, math.Exp, -103, 88.7, 1},
		{"sigmoid", sigmoid32, sigmoid64, -80, 20, 2},
		{"tanh", tanh32, math.Tanh, -10, 10, 2},
		{"tanh near 0", tanh32, math.Tanh, -1e-3, 1e-3, 2},
	}

	for _, test := range tests {
		worst := maxULPError(test.f32, test.f64, test.from, test.to, 2_000_000)
		t.Logf("%s: at most %d ulp from the correctly rounded result", test.name, worst)
		assert.LessOrEqual(t, worst, test.maxULP, test.name)
	}
}

func TestActivation32SpecialValues(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())

	assert.Equal(t, float32(1), exp32(0))
	assert.Equal(t, inf, exp32(inf))
	assert.Equal(t, inf, exp32(89))
	assert.Equal(t, float32(0), exp32(-inf))
	assert.Equal(t, float32(0), exp32(-104))
	assert.True(t, math.IsNaN(float64(exp32(nan))))

	assert.Equal(t, float32(0.5), sigmoid32(0))
	assert.Equal(t, float32(1), sigmoid32(inf))
	assert.Equal(t, float32(0), sigmoid32(-inf))
	assert.True(t, math.IsNaN(float64(sigmoid32(nan))))

	assert.Equal(t, float32(0), tanh32(0))
	assert.Equal(t, float32(1), tanh32(inf))
	assert.Equal(t, float32(-1), tanh32(-inf))
	assert.Equal(t, float32(1), tanh32(50))
	assert.True(t, math.IsNaN(float64(tanh32(nan))))
}

func BenchmarkActivation(b *testing.B) {
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i)/32 - 4
	}

	b.Run("sigmoid32", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, v := range values {
				_ = sigmoid32(v)
			}
		}
	})

	b.Run("sigmoid64", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, v := range values {
				_ = float32(1 / (1 + math.Exp(-float64(v))))
			}
		}
	})

	b.Run("tanh32", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, v := range values {
				_ = tanh32(v)
			}
		}
	})

	b.Run("tanh64", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, v := range values {
				_ = float32(math.Tanh(float64(v)))
			}
		}
	})
}

func BenchmarkActivate(b *testing.B) {
	values := make([]float32, 256)

	for i := 0; i < b.N; i++ {
		for j := range values {
			values[j] = float32(j)/32 - 4
		}

		Activate(values, ActivationSigmoid)
	}
}
