//go:build (!arm && !arm64 && !amd64) || purego

package ops

// hasSIMD reports whether a SIMD kernel can run; this platform has none.
var hasSIMD = false

const prefetchDistance = 0

func sgemvPanels(_, _, _ *float32, _, _, _ int) {
	panic("ops: no SIMD kernel on this platform")
}
