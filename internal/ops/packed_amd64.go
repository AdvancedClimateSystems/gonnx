//go:build amd64 && !purego

package ops

import "golang.org/x/sys/cpu"

// hasSIMD reports whether the AVX kernel can run.
var hasSIMD = cpu.X86.HasAVX

// prefetchDistance is unused on amd64, where the hardware prefetcher handles the
// sequential stream of weights.
const prefetchDistance = 0

// sgemvPanels computes, for every panel, out[c] = Σ_l a[l]·packed[l][c] for the
// panelWidth columns c of the panel, summing over l in ascending order. k must be a
// positive multiple of 4.
//
//go:noescape
func sgemvPanels(out, packed, a *float32, k, panels, prefetch int)
