package ops

import "math"

// The float32 activation functions below compute in float32 with the polynomial
// approximations of the Cephes math library (expf and tanhf). They take far fewer
// cycles than the float64 functions of the math package, most of all on 32-bit ARM,
// where those are not implemented in assembly.

const (
	// exp32Max and exp32Min bound the arguments for which exp32 is a finite, non-zero
	// float32.
	exp32Max = 88.72283905206835
	exp32Min = -103.27892990343184

	log2E = 1.44269504088896341

	// ln2High + ln2Low is ln(2), split so that n·ln2High is exact for the n used in
	// the range reduction of exp32.
	ln2High = 0.693359375
	ln2Low  = -2.12194440e-4

	// tanh32Saturation is the magnitude from which tanh32 is ±1 in float32.
	tanh32Saturation = 0.5 * exp32Max
)

// exp32 returns e^x.
func exp32(x float32) float32 {
	switch {
	case x != x:
		return x
	case x > exp32Max:
		return float32(math.Inf(1))
	case x < exp32Min:
		return 0
	}

	// e^x = 2^n · e^r with r in [-ln(2)/2, ln(2)/2].
	z := x*log2E + 0.5

	n := int32(z)
	if float32(n) > z {
		n--
	}

	r := x - float32(n)*ln2High
	r -= float32(n) * ln2Low

	p := ((((1.9875691500e-4*r+1.3981999507e-3)*r+8.3334519073e-3)*r+4.1665795894e-2)*r+1.6666665459e-1)*r + 5.0000001201e-1
	y := p*r*r + r + 1

	return scaleByPowerOfTwo(y, n)
}

// scaleByPowerOfTwo returns y·2^n for a y near 1 and n in [-150, 128].
func scaleByPowerOfTwo(y float32, n int32) float32 {
	if n > 127 {
		y *= 2
		n--
	}

	if n < -126 {
		y *= math.Float32frombits(1 << 23) // 2^-126
		n += 126
	}

	return y * math.Float32frombits(uint32(n+127)<<23)
}

// sigmoid32 returns 1 / (1 + e^-x).
func sigmoid32(x float32) float32 {
	return 1 / (1 + exp32(-x))
}

// tanh32 returns the hyperbolic tangent of x.
func tanh32(x float32) float32 {
	z := x
	if z < 0 {
		z = -z
	}

	switch {
	case z > tanh32Saturation:
		if x < 0 {
			return -1
		}

		return 1
	case z >= 0.625:
		// For larger arguments, tanh(z) = 1 - 2 / (e^2z + 1) loses no precision.
		y := 1 - 2/(exp32(z+z)+1)
		if x < 0 {
			return -y
		}

		return y
	case x == 0:
		return x
	default:
		s := x * x

		return ((((-5.70498872745e-3*s+2.06390887954e-2)*s-5.37397155531e-2)*s+1.33314422036e-1)*s-3.33332819422e-1)*s*x + x
	}
}
