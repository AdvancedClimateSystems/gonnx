//go:build arm64 && !purego

package ops

// hasSIMD reports whether the NEON kernel can run; every arm64 processor has NEON.
var hasSIMD = true

// prefetchDistance is how many bytes ahead of its loads the kernel prefetches weights.
const prefetchDistance = 256

// sgemvPanels computes, for every panel, out[c] = Σ_l a[l]·packed[l][c] for the
// panelWidth columns c of the panel, summing over l in ascending order. k must be a
// positive multiple of 4.
//
//go:noescape
func sgemvPanels(out, packed, a *float32, k, panels, prefetch int)
