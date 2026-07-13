// ═══════════════════════════════════════════════════════════════
// prefill.hpp — S-MoE Engine · Layer-Major Batched Prefill
// ═══════════════════════════════════════════════════════════════
// Token-serial prefill reads every routed expert once PER TOKEN
// (~7 GB of Direct I/O per prompt token on Qwen3-235B). Layer-major
// prefill walks the prompt chunk-by-chunk, LAYER by layer: all
// tokens of the chunk pass through layer L before L+1, and the
// chunk's routed experts are deduplicated so each expert blob is
// claimed ONCE per layer per chunk and applied to every token that
// routed to it.
//
// I/O bound drops from  count × layers × 8  blob reads to
// Σ_layers |union(chunk routing)| ≤ min(experts_per_layer, 8×chunk),
// and the whole union is prefetched as soon as the layer's routing
// is known, so the streamer workers run ahead of consumption.
//
// Invariants (CLAUDE.md): all buffers pre-allocated by the caller,
// 16 KB-aligned and Metal-registered; zero heap allocation inside
// run_layer_major(); synchronisation via the streamer's atomics only.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "common.hpp"
#include "io/streamer.hpp"
#include "scout/scout.hpp"
#include "compute/metal_bridge.h"

namespace smoe::prefill {

// Tokens processed per chunk. Bounds activation-plane memory
// (CHUNK × d_model floats per plane) and the per-layer union size.
inline constexpr uint32_t CHUNK = 64;

// Byte offsets/shapes of the three quantised tensors inside one expert
// blob, read from the vault's TensorDescriptor table (see main.cpp
// read_expert_layout).
struct ExpertLayout {
    uint32_t gate_rows;
    uint32_t gate_cols;
    uint64_t gate_packed_offset;
    uint64_t gate_scales_offset;

    uint32_t up_rows;
    uint32_t up_cols;
    uint64_t up_packed_offset;
    uint64_t up_scales_offset;

    uint32_t down_rows;
    uint32_t down_cols;
    uint64_t down_packed_offset;
    uint64_t down_scales_offset;
};

// Pre-allocated activation planes (row i = chunk token i).
// All Metal-touched planes must be posix_memalign(16 KB) and
// registered via smoe_metal_register_buffer by the caller.
struct Buffers {
    float* hidden;     // [CHUNK × d_model]
    float* normed;     // [CHUNK × d_model]
    float* qbuf;       // [CHUNK × q_dim]
    float* kbuf;       // [CHUNK × kv_dim]
    float* vbuf;       // [CHUNK × kv_dim]
    float* attn_out;   // [CHUNK × q_dim]
    float* routed_out; // [CHUNK × d_model]  (CPU accumulator)
    float* oproj_out;  // [CHUNK × d_model]  (o_proj rows, batched dispatch)
    float* ffn_hidden; // [ffn_dim]          (expert / dense gate scratch)
    float* ffn_up;     // [ffn_dim]          (dense up scratch)
    float* ffn_out;    // [d_model]          (expert / dense down output)
    float* sh_gate;    // [shared_dim]       (shared-expert scratch, may be null)
    float* sh_up;      // [shared_dim]
    float* sh_out;     // [d_model]
    float* gate_scores;// [max_experts_per_layer] (CPU)
    float* rope_cos;   // [CHUNK × head_dim/2]    (CPU)
    float* rope_sin;   // [CHUNK × head_dim/2]    (CPU)
    float* batch_in;   // [CHUNK × d_model] token-batch FFN input staging (CPU)
    float* batch_out;  // [CHUNK × d_model] token-batch FFN output staging (CPU)
};

// Stride of the expert-hit histogram: hits[layer * HITS_STRIDE + expert].
inline constexpr uint32_t HITS_STRIDE = 512;

struct Params {
    const SmoeModelConfig*  cfg;
    smoe::scout::Scout*     scout;
    smoe::io::Streamer*     streamer;
    SmoeMetalCtx*           metal;
    const SmoeHeader*       hdr;
    const ExpertLayout*     layout;
    float*                  full_kv_cache; // [layers][2][attn_ctx][kv_dim]
    uint32_t                attn_ctx;
    uint32_t                moe_start_layer;
    const float*            rope_inv_freq; // [head_dim/2]
    Buffers                 b;
    uint32_t*               expert_hits;   // optional popularity histogram
    bool                    debug;
};

// Run layer-major prefill for tokens[0..count) whose absolute stream
// positions start at stream_base. Advances ctx_pos/ctx_fill by count
// and mirrors K/V into the scout's cache (scout position itself is
// synced separately by the caller at the generation boundary).
// attn_scores: caller-provided [attn_ctx] scratch.
void run_layer_major(const Params& P,
                     const uint32_t* tokens, uint32_t count,
                     uint64_t stream_base,
                     uint32_t& ctx_pos, uint32_t& ctx_fill,
                     float* attn_scores);

} // namespace smoe::prefill
