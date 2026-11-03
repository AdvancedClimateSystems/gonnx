//go:build amd64 && !purego

#include "textflag.h"

// func sgemvPanels(out, packed, a *float32, k, panels, prefetch int)
//
// Panels are processed in pairs so four accumulators (Y8-Y11) are in flight, which hides
// the latency of the additions. Per column l, the value a[l] is broadcast, multiplied
// with the weights of that column and then added, with the same rounding as the separate
// multiplications and additions Go emits for the scalar code.
TEXT ·sgemvPanels(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ a+16(FP), DX
	MOVQ k+24(FP), CX
	MOVQ panels+32(FP), BX
	MOVQ CX, R10
	SHLQ $6, R10 // bytes per panel: k columns of 16 float32 weights

pair:
	CMPQ BX, $2
	JLT  single
	LEAQ (SI)(R10*1), R11 // second panel of the pair
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
	MOVQ   DX, R8
	MOVQ   CX, R9

pairInner:
	VBROADCASTSS (R8), Y0
	VMULPS       (SI), Y0, Y1
	VMULPS       32(SI), Y0, Y2
	VMULPS       (R11), Y0, Y3
	VMULPS       32(R11), Y0, Y4
	VADDPS       Y1, Y8, Y8
	VADDPS       Y2, Y9, Y9
	VADDPS       Y3, Y10, Y10
	VADDPS       Y4, Y11, Y11
	ADDQ         $4, R8
	ADDQ         $64, SI
	ADDQ         $64, R11
	DECQ         R9
	JNZ          pairInner

	VMOVUPS Y8, (DI)
	VMOVUPS Y9, 32(DI)
	VMOVUPS Y10, 64(DI)
	VMOVUPS Y11, 96(DI)
	ADDQ    $128, DI
	MOVQ    R11, SI
	SUBQ    $2, BX
	JMP     pair

single:
	TESTQ BX, BX
	JZ    done
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	MOVQ   DX, R8
	MOVQ   CX, R9

singleInner:
	VBROADCASTSS (R8), Y0
	VMULPS       (SI), Y0, Y1
	VMULPS       32(SI), Y0, Y2
	VADDPS       Y1, Y8, Y8
	VADDPS       Y2, Y9, Y9
	ADDQ         $4, R8
	ADDQ         $64, SI
	DECQ         R9
	JNZ          singleInner

	VMOVUPS Y8, (DI)
	VMOVUPS Y9, 32(DI)

done:
	VZEROUPPER
	RET
