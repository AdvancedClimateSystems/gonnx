package ops

import "github.com/advancedclimatesystems/gonnx/tensor"

// panelWidth is the number of outputs the SIMD kernels compute at once: a packed matrix
// interleaves the weights of that many outputs.
const panelWidth = 16

// PackedMatrix holds the float32 weights B of a product A·Bᵀ, with B of size n×k, in the
// layout the SIMD kernel of this platform reads as one sequential stream: panels of
// panelWidth rows of B, stored column by column. The last panel is padded with zeros.
// The kernels process columns in groups of four; the remaining k%4 columns are kept
// row by row in tail.
type PackedMatrix struct {
	panels []float32
	tail   []float32
	n, k   int
}

// PackTransposed packs B given as an n×k row-major matrix. It returns nil when this
// platform has no SIMD kernel, in which case callers use MatMulInto instead.
func PackTransposed(b []float32, n, k int) *PackedMatrix {
	return pack(n, k, func(j, l int) float32 { return b[j*k+l] })
}

// Pack packs B given as its transpose: a k×n row-major matrix. It returns nil when this
// platform has no SIMD kernel, in which case callers use MatMulInto instead.
func Pack(b []float32, k, n int) *PackedMatrix {
	return pack(n, k, func(j, l int) float32 { return b[l*n+j] })
}

func pack(n, k int, element func(j, l int) float32) *PackedMatrix {
	if !hasSIMD {
		return nil
	}

	vectorLength := k &^ 3
	nPanels := (n + panelWidth - 1) / panelWidth
	p := &PackedMatrix{
		panels: make([]float32, nPanels*vectorLength*panelWidth),
		tail:   make([]float32, n*(k-vectorLength)),
		n:      n,
		k:      k,
	}

	for j := 0; j < n; j++ {
		panel, column := j/panelWidth, j%panelWidth

		for l := 0; l < vectorLength; l++ {
			p.panels[(panel*vectorLength+l)*panelWidth+column] = element(j, l)
		}

		for l := vectorLength; l < k; l++ {
			p.tail[j*(k-vectorLength)+l-vectorLength] = element(j, l)
		}
	}

	return p
}

// Dims returns the size n×k of B.
func (p *PackedMatrix) Dims() (int, int) {
	return p.n, p.k
}

// MatMul computes out = a·Bᵀ, where a is an m×k and out an m×n row-major matrix. Like
// MatMulInto, every output element is summed over k in ascending order.
func (p *PackedMatrix) MatMul(out, a []float32, m int) {
	for i := 0; i < m; i++ {
		p.matVec(out[i*p.n:][:p.n], a[i*p.k:][:p.k])
	}
}

func (p *PackedMatrix) matVec(out, a []float32) {
	vectorLength := p.k &^ 3
	fullPanels := p.n / panelWidth
	remainder := p.n % panelWidth

	if vectorLength == 0 {
		for j := range out {
			out[j] = 0
		}
	} else {
		if fullPanels > 0 {
			sgemvPanels(&out[0], &p.panels[0], &a[0], vectorLength, fullPanels, prefetchDistance)
		}

		if remainder > 0 {
			var last [panelWidth]float32

			sgemvPanels(&last[0], &p.panels[fullPanels*vectorLength*panelWidth], &a[0], vectorLength, 1, prefetchDistance)
			copy(out[fullPanels*panelWidth:], last[:remainder])
		}
	}

	if tailLength := p.k - vectorLength; tailLength > 0 {
		for j := range out {
			sum := out[j]
			row := p.tail[j*tailLength:][:tailLength]

			for l, value := range row {
				sum += a[vectorLength+l] * value
			}

			out[j] = sum
		}
	}
}

// minPackedSize is the number of weights from which packing pays off. Products with
// smaller matrices are dominated by the cost of calling the kernel, and most of a panel
// would be padding.
const minPackedSize = 1024

// PreparedWeights is a constant float32 weight tensor together with its packed form.
type PreparedWeights struct {
	source *tensor.Tensor
	packed *PackedMatrix
}

// PrepareWeights packs a float32 weight tensor for products A·Bᵀ. With transposed set,
// the tensor holds B with the last dimension of length k; otherwise it holds the k×n
// matrix Bᵀ. It returns nil when t is nil, does not fit, is too small to benefit or
// cannot be packed on this platform.
func PrepareWeights(t *tensor.Tensor, transposed bool) *PreparedWeights {
	if t == nil || t.DataType() != tensor.Float32 || t.Rank() < 2 || t.Size() < minPackedSize {
		return nil
	}

	var packed *PackedMatrix

	data := tensor.Data[float32](t)

	switch {
	case transposed:
		k := t.Shape()[t.Rank()-1]
		packed = PackTransposed(data, t.Size()/k, k)
	case t.Rank() == 2:
		packed = Pack(data, t.Shape()[0], t.Shape()[1])
	}

	if packed == nil {
		return nil
	}

	return &PreparedWeights{source: t, packed: packed}
}

// For returns the packed weights when t is the tensor they were prepared from, and nil
// otherwise.
func (w *PreparedWeights) For(t *tensor.Tensor) *PackedMatrix {
	if w == nil || w.source != t {
		return nil
	}

	return w.packed
}
