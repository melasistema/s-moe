// ═══════════════════════════════════════════════════════════════
// metal_bridge.h — S-MoE Engine · Apple Metal GPU Interface
// ═══════════════════════════════════════════════════════════════
// Phase 3 — Week 3
//
// Pure C interface so it can be included from C++20 translation
// units (streamer.cpp, main.cpp) without dragging in ObjC headers.
//
// The Metal bridge manages:
//   • JIT compilation of the MSL shader from embedded source string.
//   • A ping-pong pair of MTLBuffers in Unified Memory (Shared).
//   • Zero-copy pointer injection: pread() writes directly into the
//     MTLBuffer's backing pages; the GPU reads the same pages.
//   • Dispatch of the two-pass SMOE-Q2 FFN kernel pipeline.
//   • Atomic buffer-swap handshake with the I/O streamer.
//
// Invariants:
//   • GPU executes Buffer A while I/O is filling Buffer B.
//   • Swap is a plain pointer exchange — no data copy ever occurs.
//   • All synchronisation is via atomic flags, never OS mutexes.
// ═══════════════════════════════════════════════════════════════

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Opaque handle to the Metal execution context.
typedef struct SmoeMetalCtx SmoeMetalCtx;

// ── Lifecycle ────────────────────────────────────────────────

// Initialise Metal device, command queue, JIT-compile the
// SMOE-Q2 MSL shader, and allocate the ping-pong UMA buffers.
//
//   slot_bytes  — size of each ping-pong buffer in bytes.
//                 Must be a multiple of PAGE_SIZE (16 KB).
//                 Typically: max_expert_padded_size.
//
// Returns NULL on failure (device unavailable, compilation error).
SmoeMetalCtx* smoe_metal_init(size_t slot_bytes);

// Release all Metal resources. Safe to call with NULL.
void smoe_metal_destroy(SmoeMetalCtx* ctx);

// ── Buffer access ─────────────────────────────────────────────

// Returns a CPU-writable pointer into the *passive* (B) buffer.
// The I/O streamer calls pread() directly into this address.
// The pointer is stable until the next smoe_metal_swap_buffers().
void* smoe_metal_passive_ptr(SmoeMetalCtx* ctx);

// Returns the size of each ping-pong buffer in bytes.
size_t smoe_metal_slot_bytes(const SmoeMetalCtx* ctx);

// ── Compute dispatch ──────────────────────────────────────────

// Submit a two-pass FFN compute pipeline on the *active* (A) buffer.
//
// The function encodes both passes (smoe_gate_up + smoe_down) into
// a single command buffer and commits it synchronously
// (waitUntilCompleted). For the current validation build this is
// sufficient; async dispatch is Phase 4 territory.
//
//   expert_blob  — pointer to the active buffer's contents
//                  (layout: gate blob | up blob | down blob, each
//                   described by the TensorDescriptor trio in common.hpp)
//   gate_td,
//   up_td,
//   down_td      — tensor descriptors from the .smoe vault for this
//                  expert (must match the blob's internal layout)
//   input_vec    — float32 activation input  (cols elements)
//   output_vec   — float32 activation output (rows elements)
//   rows         — number of rows (intermediate / output dim)
//   cols         — number of cols (input dim)
//   group_size   — SMOE-Q2 group size (always 64)
void smoe_metal_fused_ffn(SmoeMetalCtx*   ctx,
                           const uint8_t*  packed_gate,
                           const uint16_t* scales_gate,
                           const uint8_t*  packed_up,
                           const uint16_t* scales_up,
                           const uint8_t*  packed_down,
                           const uint16_t* scales_down,
                           const float*    input_vec,
                           float*          hidden_vec,   // caller-allocated scratch
                           float*          output_vec,
                           uint32_t        rows,
                           uint32_t        cols,
                           uint32_t        group_size,
                           uint32_t        bits);

// Perform a float32 matrix-vector multiplication on the GPU for Scout projections.
void smoe_metal_scout_matvec(SmoeMetalCtx* ctx,
                             const float*   weight,
                             const float*   input_vec,
                             float*         output_vec,
                             uint32_t       rows,
                             uint32_t       cols);

// Perform a batch of float32 matrix-vector multiplications on the GPU to reduce sync latency.
void smoe_metal_scout_matvec_batch(SmoeMetalCtx* ctx,
                                   const float**  weights,
                                   const float**  inputs,
                                   float**        outputs,
                                   const uint32_t* rows,
                                   const uint32_t* cols,
                                   uint32_t       count);

// Register a CPU buffer with Metal once at boot to avoid page table mapping overhead in the hot loop.
void smoe_metal_register_buffer(SmoeMetalCtx* ctx, const void* ptr, size_t size_in_bytes);




// ── Ping-pong control ─────────────────────────────────────────

// Atomically swap the active (A) and passive (B) buffers.
// Call after smoe_metal_fused_ffn returns and the output has been
// consumed. The next smoe_metal_passive_ptr() call will return the
// old A buffer — the I/O streamer can now overwrite it.
void smoe_metal_swap_buffers(SmoeMetalCtx* ctx);

// Returns true if the passive buffer is ready to be filled
// (i.e. has been fully consumed by the GPU and swapped out).
bool smoe_metal_passive_ready(const SmoeMetalCtx* ctx);

// ── Telemetry ─────────────────────────────────────────────────
uint64_t smoe_metal_kernel_dispatches(const SmoeMetalCtx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif
