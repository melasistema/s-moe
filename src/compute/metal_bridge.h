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
// Returns a handle to wait on.
void* smoe_metal_fused_ffn(SmoeMetalCtx*   ctx,
                           const uint8_t*  packed_gate,
                           const uint16_t* scales_gate,
                           const uint8_t*  packed_up,
                           const uint16_t* scales_up,
                           const uint8_t*  packed_down,
                           const uint16_t* scales_down,
                           const float*    input_vec,
                           float*          hidden_vec,   // caller-allocated scratch
                           float*          output_vec,
                           uint32_t        gate_rows,
                           uint32_t        gate_cols,
                           uint32_t        down_rows,
                           uint32_t        down_cols,
                           uint32_t        group_size,
                           uint32_t        bits,
                           uint32_t        expert_index);

void smoe_metal_wait(SmoeMetalCtx* ctx, void* handle, float* output_vec, uint32_t expert_index, uint32_t cols);

// ── Grouped expert dispatch (decode hot loop) ─────────────────
// Encodes N same-shaped experts into ONE command buffer:
// all gate_up passes, one barrier, all down passes, one signal.
// Replaces N per-expert command buffers (commit/schedule overhead
// ~0.1–0.3 ms each) with one. The layer input is copied into the
// shared input buffer once per call — every expert of a layer
// consumes the same normed hidden state.
//
// `slot` selects the expert's private region in the pre-allocated
// hidden/output scratch (same indexing as smoe_metal_fused_ffn's
// expert_index). Fetch each expert's result with smoe_metal_wait
// using the returned handle and its slot.
typedef struct {
    const uint8_t*  packed_gate;
    const uint16_t* scales_gate;
    const uint8_t*  packed_up;
    const uint16_t* scales_up;
    const uint8_t*  packed_down;
    const uint16_t* scales_down;
    uint32_t        slot;
} SmoeExpertBlob;

void* smoe_metal_fused_ffn_group(SmoeMetalCtx*        ctx,
                                 const SmoeExpertBlob* experts,
                                 uint32_t              n_experts,
                                 const float*          input_vec,
                                 uint32_t              gate_rows,
                                 uint32_t              gate_cols,
                                 uint32_t              down_rows,
                                 uint32_t              down_cols,
                                 uint32_t              group_size,
                                 uint32_t              bits);

// ── Token-batch fused FFN (layer-major prefill) ──────────────
// Applies ONE expert to `batch` token activations in a single
// two-pass dispatch. input_mat is row-major [batch × gate_cols];
// results are fetched with smoe_metal_wait_ffn_batch into
// out_mat [batch × down_rows]. Returns NULL if the batch/dims
// exceed the pre-allocated staging buffers (caller must fall back
// to the per-token path).
#define SMOE_MAX_FFN_BATCH 64
#define SMOE_MAX_FFN_BATCH_DIM 16384

void* smoe_metal_fused_ffn_batch(SmoeMetalCtx*   ctx,
                                 const uint8_t*  packed_gate,
                                 const uint16_t* scales_gate,
                                 const uint8_t*  packed_up,
                                 const uint16_t* scales_up,
                                 const uint8_t*  packed_down,
                                 const uint16_t* scales_down,
                                 const float*    input_mat,
                                 uint32_t        batch,
                                 uint32_t        gate_rows,
                                 uint32_t        gate_cols,
                                 uint32_t        down_rows,
                                 uint32_t        down_cols,
                                 uint32_t        group_size,
                                 uint32_t        bits);

void smoe_metal_wait_ffn_batch(SmoeMetalCtx* ctx, void* handle,
                               float* out_mat, uint32_t batch, uint32_t cols);

// Perform a float32 matrix-vector multiplication on the GPU for Scout projections.
void smoe_metal_scout_matvec(SmoeMetalCtx* ctx, const float* weight, const float* input, float* output, uint32_t rows, uint32_t cols);
void smoe_metal_scout_matvec_bf16(SmoeMetalCtx* ctx,
                             const uint16_t* weight,
                             const float*   input_vec,
                             float*         output_vec,
                             uint32_t       rows,
                             uint32_t       cols);

// ── Fused decode attention layer (single token) ───────────────
// One command buffer, one CPU sync: QKV projections → per-head
// QK-RMSNorm + RoPE + KV ring append → GQA attention → O projection.
// Replaces the previous two synchronous matvec roundtrips per layer
// (QKV, then o_proj after CPU attention) and moves the O(context)
// attention math onto the GPU. Synchronous: o_out is valid on return.
// NOT bit-exact with the CPU path (simd-reduced norms/softmax, GPU
// exp) — verify token-level, not bit-level.
typedef struct {
    const uint16_t* w_q;        // [q_dim × d_model] bf16
    const uint16_t* w_k;        // [kv_dim × d_model] bf16
    const uint16_t* w_v;        // [kv_dim × d_model] bf16
    const uint16_t* w_o;        // [d_model × q_dim] bf16
    const uint16_t* q_norm_w;   // [head_dim] bf16, NULL → no QK norm
    const uint16_t* k_norm_w;   // [head_dim] bf16, NULL → no QK norm
    const float*    rope_cos;   // [head_dim/2] for this token's position
    const float*    rope_sin;   // [head_dim/2]
    const float*    normed_in;  // [d_model] post input-norm hidden
    float*          qbuf;       // [q_dim] scratch (normed+roped Q on exit)
    float*          kbuf;       // [kv_dim] scratch
    float*          vbuf;       // [kv_dim] scratch
    float*          k_cache;    // layer base of KV ring [attn_ctx × kv_dim]
    float*          v_cache;    // layer base of KV ring [attn_ctx × kv_dim]
    float*          attn_out;   // [q_dim] scratch
    float*          o_out;      // [d_model] result
    uint32_t d_model, q_dim, kv_dim, head_dim, num_heads, num_kv_heads;
    uint32_t slot;              // KV ring write slot (ctx_pos % attn_ctx)
    uint32_t valid;             // attended positions incl. current
    uint32_t attn_ctx;          // KV ring capacity
    float    attn_scale;        // 1/sqrt(head_dim)
} SmoeAttnLayerArgs;

void smoe_metal_attention_layer(SmoeMetalCtx* ctx, const SmoeAttnLayerArgs* args);

// Perform a batch of float32 matrix-vector multiplications on the GPU to reduce sync latency.
void smoe_metal_scout_matvec_batch(SmoeMetalCtx* ctx, const float** weights, const float** inputs, float** outputs, const uint32_t* rows, const uint32_t* cols, uint32_t count);
void smoe_metal_scout_matvec_batch_bf16(SmoeMetalCtx* ctx,
                                   const uint16_t** weights,
                                   const float**  inputs,
                                   float**        outputs,
                                   const uint32_t* rows,
                                   const uint32_t* cols,
                                   uint32_t       count);

// Same as smoe_metal_scout_matvec_batch_bf16 but WITHOUT the memory
// barrier between dispatches: the caller guarantees every output plane
// is distinct and no dispatch reads another's output, so the GPU is
// free to overlap them (layer-major prefill: all B tokens' QKV in one
// command buffer, then all B o_proj rows in another).
void smoe_metal_scout_matvec_group_bf16(SmoeMetalCtx* ctx,
                                   const uint16_t** weights,
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
