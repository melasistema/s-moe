// ═══════════════════════════════════════════════════════════════
// decode.hpp — S-MoE Engine · Token-Serial Decode Step
// ═══════════════════════════════════════════════════════════════
// The twin of prefill.hpp: where layer-major prefill pushes a whole
// prompt chunk through each layer once, decode pushes ONE token
// through all layers — exact router-gate evaluation on the heavy
// hidden state, demand prefetch of the routed experts, speculative
// prefetch of the deeper gate ranks, grouped GPU dispatch, and a
// spin-wait that turns NVMe completions into command buffers.
//
// One call = one token step: embedding → every layer's attention +
// FFN → KV ring advance. The final hidden state is left in the
// caller's buffer for the LM head + sampler (which stay with the
// caller, so sampling/emission live in one place).
//
// Invariants (CLAUDE.md): all buffers pre-allocated by the caller
// (engine::Buffers), zero heap allocation inside run_token_step();
// synchronisation via the streamer's atomics only.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "common.hpp"
#include "engine.hpp"
#include "io/streamer.hpp"
#include "prefill.hpp"
#include "scout/scout.hpp"
#include "compute/metal_bridge.h"

namespace smoe::decode {

// KV ring capacity in tokens. Fixed at compile time: the KV cache,
// attention scratch, and the serve-layer context meter all share it.
inline constexpr uint32_t ATTN_CTX = 4096;

// ── --instrument state ────────────────────────────────────────
// Wall-time buckets over the serial generation loop, accumulated per
// request and printed at request end. Buckets are contiguous segments
// between marks on the calling thread, so their sum plus "other"
// equals the measured token total — no double counting.
struct Instr {
    double   dense_ms    { 0.0 };  // norms + QKV + attention + o_proj (+ shared)
    double   dispatch_ms { 0.0 };  // Phase 1 first claim + GPU dispatch pass
    double   io_spin_ms  { 0.0 };  // Phase 3 spin-wait on missing experts
    double   gpu_wait_ms { 0.0 };  // Phase 4 kernel wait + accumulate
    double   lm_ms       { 0.0 };  // final norm + LM head + sampling
    double   total_ms    { 0.0 };
    uint64_t tokens      { 0 };
    uint64_t bytes       { 0 };
    uint64_t ovl_hits    { 0 };    // experts shared with the previous token
    uint64_t ovl_total   { 0 };    // experts routed this token
    uint64_t cov[4]      { 0, 0, 0, 0 };  // true experts inside prev token's top-8/16/24/32
    uint64_t cov_total   { 0 };

    double   mark        { 0.0 };  // last bucket boundary (wall_ms)

    // Previous token's routed-expert bitmask per layer, for the
    // adjacent-token overlap measurement (≤128 experts → two words),
    // and its top-32 gate ranking, for prefetch-width coverage.
    uint64_t prev_mask[128][2] {};
    uint32_t prev_top32[128][32] {};
    uint8_t  prev_top32_n[128] {};

    // Charge the wall time since the previous mark to one bucket.
    void bucket(double& acc) noexcept {
        const double now = smoe::engine::wall_ms();
        acc += now - mark;
        mark = now;
    }

    // Print the per-request breakdown on stderr and reset the buckets
    // (coverage/overlap state survives — it spans requests).
    void print_and_reset();
};

// ── Per-boot decode parameters (mirrors prefill::Params) ──────
struct Params {
    const SmoeModelConfig*             cfg;
    smoe::scout::Scout*                scout;
    smoe::io::Streamer*                streamer;
    SmoeMetalCtx*                      metal;
    const SmoeHeader*                  hdr;
    const smoe::prefill::ExpertLayout* layout;
    const smoe::engine::Buffers*       b;
    uint32_t                           moe_start_layer;
    uint32_t                           spec_width;   // extra speculative gate ranks
    uint32_t*                          expert_hits;  // popularity histogram
    bool                               debug;
};

// Run ONE token through all layers: embedding → attention (fused GPU
// path or CPU fallback) → exact routing → routed+shared FFN → residual,
// then advance the KV ring. The final (un-normed) hidden state is left
// in P.b->hidden for the caller's LM head + sampler.
//
//   in         — instrumentation state; nullptr when --instrument is off.
//                Buckets are charged only when !is_prompt.
//   token      — the token id this step processes
//   stream_pos — ABSOLUTE stream position (RoPE must not use request-
//                relative indices: serve requests start mid-stream)
//   is_prompt  — prompt step (seeds instr coverage without counting)
//   first_step — first step of this request (debug prints only)
void run_token_step(const Params& P, Instr* in,
                    uint32_t token, uint64_t stream_pos,
                    bool is_prompt, bool first_step,
                    uint32_t& ctx_pos, uint32_t& ctx_fill);

} // namespace smoe::decode
