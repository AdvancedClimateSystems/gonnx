//go:build arm && !purego

#include "textflag.h"

// func sgemvPanels(out, packed, a *float32, k, panels, prefetch int)
//
// Go's arm assembler has no NEON instructions, so they are encoded as WORDs, each
// annotated with the instruction it encodes (ARMv7-A, A1 encoding). Each panel
// accumulates its 16 outputs in q8-q11. Per column l, the value a[l] is multiplied with
// the 16 weights of that column by VMLA, which rounds the product before adding it,
// like the separate multiplications and additions Go emits for the scalar code. The
// weights are prefetched prefetch bytes ahead with PLD.
TEXT ·sgemvPanels(SB), NOSPLIT, $0-24
	MOVW out+0(FP), R0
	MOVW packed+4(FP), R1
	MOVW a+8(FP), R2
	MOVW k+12(FP), R3
	MOVW panels+16(FP), R4
	MOVW prefetch+20(FP), R7
	ADD $32, R7, R8
panel:
	WORD $0xF2C00050 // VMOV.I32 q8, #0
	WORD $0xF2C02050 // VMOV.I32 q9, #0
	WORD $0xF2C04050 // VMOV.I32 q10, #0
	WORD $0xF2C06050 // VMOV.I32 q11, #0
	MOVW R2, R5
	MOVW R3, R6
inner:
	WORD $0xF4250A8D // VLD1.32 {d0-d1}, [R5]!
	WORD $0xF421428D // VLD1.32 {d4-d7}, [R1]!
	WORD $0xF421828D // VLD1.32 {d8-d11}, [R1]!
	WORD $0xF7D1F007 // PLD [R1, R7]
	WORD $0xF7D1F008 // PLD [R1, R8]
	WORD $0xF3E40140 // VMLA.F32 q8, q2, d0[0]
	WORD $0xF3E62140 // VMLA.F32 q9, q3, d0[0]
	WORD $0xF3E84140 // VMLA.F32 q10, q4, d0[0]
	WORD $0xF3EA6140 // VMLA.F32 q11, q5, d0[0]
	WORD $0xF461828D // VLD1.32 {d24-d27}, [R1]!
	WORD $0xF461C28D // VLD1.32 {d28-d31}, [R1]!
	WORD $0xF7D1F007 // PLD [R1, R7]
	WORD $0xF7D1F008 // PLD [R1, R8]
	WORD $0xF3E801E0 // VMLA.F32 q8, q12, d0[1]
	WORD $0xF3EA21E0 // VMLA.F32 q9, q13, d0[1]
	WORD $0xF3EC41E0 // VMLA.F32 q10, q14, d0[1]
	WORD $0xF3EE61E0 // VMLA.F32 q11, q15, d0[1]
	WORD $0xF421428D // VLD1.32 {d4-d7}, [R1]!
	WORD $0xF421828D // VLD1.32 {d8-d11}, [R1]!
	WORD $0xF7D1F007 // PLD [R1, R7]
	WORD $0xF7D1F008 // PLD [R1, R8]
	WORD $0xF3E40141 // VMLA.F32 q8, q2, d1[0]
	WORD $0xF3E62141 // VMLA.F32 q9, q3, d1[0]
	WORD $0xF3E84141 // VMLA.F32 q10, q4, d1[0]
	WORD $0xF3EA6141 // VMLA.F32 q11, q5, d1[0]
	WORD $0xF461828D // VLD1.32 {d24-d27}, [R1]!
	WORD $0xF461C28D // VLD1.32 {d28-d31}, [R1]!
	WORD $0xF7D1F007 // PLD [R1, R7]
	WORD $0xF7D1F008 // PLD [R1, R8]
	WORD $0xF3E801E1 // VMLA.F32 q8, q12, d1[1]
	WORD $0xF3EA21E1 // VMLA.F32 q9, q13, d1[1]
	WORD $0xF3EC41E1 // VMLA.F32 q10, q14, d1[1]
	WORD $0xF3EE61E1 // VMLA.F32 q11, q15, d1[1]
	SUB.S $4, R6
	BNE inner
	WORD $0xF440028D // VST1.32 {d16-d19}, [R0]!
	WORD $0xF440428D // VST1.32 {d20-d23}, [R0]!
	SUB.S $1, R4
	BNE panel
	RET
