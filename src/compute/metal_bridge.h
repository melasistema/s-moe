// ═══════════════════════════════════════════════════════════════
// metal_bridge.h — S-MoE Engine · Apple Metal GPU Interface
// ═══════════════════════════════════════════════════════════════
// Phase 3 stub — to be implemented in Week 3.
//
// The Metal bridge manages:
//   • A ping-pong pair of MTLBuffers (Buffer A, Buffer B).
//   • Zero-copy pointer injection from UMA into the GPU.
//   • Scheduling of the 2-bit dequantising FFN kernel.
//
// Invariant: GPU executes Buffer A while I/O fills Buffer B.
//            Pointer swap is atomic; no copy ever occurs.
// ═══════════════════════════════════════════════════════════════

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handle to the Metal execution context.
typedef struct SmoeMetalCtx SmoeMetalCtx;

// Initialise Metal device, command queue, and ping-pong buffers.
// slot_bytes: size of each ping-pong buffer (must be PAGE_SIZE multiple).
SmoeMetalCtx* smoe_metal_init(size_t slot_bytes);

// Submit an FFN compute pass on the currently active buffer.
// packed_data  : pointer into Unified Memory holding 2-bit weights.
// scales_data  : pointer into Unified Memory holding float16 scales.
// input_vec    : activation input (float32, hidden_dim elements).
// output_vec   : destination (float32, out_dim elements).
// rows, cols   : weight matrix dimensions.
// group_size   : SMOE-Q2 group size (= 64).
void smoe_metal_fused_ffn(SmoeMetalCtx* ctx,
                           const uint8_t*  packed_data,
                           const uint16_t* scales_data,
                           const float*    input_vec,
                           float*          output_vec,
                           uint32_t rows,
                           uint32_t cols,
                           uint32_t group_size);

// Swap the active/passive buffers (ping ↔ pong).
void smoe_metal_swap_buffers(SmoeMetalCtx* ctx);

// Release all Metal resources.
void smoe_metal_destroy(SmoeMetalCtx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif
