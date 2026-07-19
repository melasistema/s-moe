// ═══════════════════════════════════════════════════════════════
// engine.hpp — S-MoE Engine · Boot-Time Machinery
// ═══════════════════════════════════════════════════════════════
// Everything main() needs before (and between) requests, none of it
// on the token hot path:
//   • slot/ring auto-tuning from the vault and available RAM
//   • the pre-allocated, Metal-registered activation buffers
//   • vocab table and expert-popularity histogram persistence
//   • the idle-time prewarm thread
//   • process telemetry primitives (wall clock, RSS)
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "common.hpp"
#include "prefill.hpp"
#include "io/streamer.hpp"
#include "compute/metal_bridge.h"

#include <atomic>
#include <string>
#include <thread>

namespace smoe::engine {

// Data files the engine persists/loads, relative to the working dir.
inline constexpr const char* VOCAB_BIN_PATH   = "vault/vocab.bin";
inline constexpr const char* EXPERT_FREQ_PATH = "vault/expert_freq.bin";

// Wall-clock time in milliseconds since an arbitrary epoch.
[[nodiscard]] double wall_ms() noexcept;

// Process resident set size in bytes (macOS task_info).
[[nodiscard]] uint64_t resident_bytes() noexcept;

// ── Slot / ring auto-tune ─────────────────────────────────────
// Fills in slot_mb and ring_size only where they are 0 (= auto):
// slot from the vault's largest expert blob, ring from a subtractive
// RAM budget (see engine.cpp for the full rationale).
void autotune(const char* vault_path, const char* scout_path,
              const SmoeHeader& hdr,
              uint64_t& slot_mb, uint32_t& ring_size);

// ── Vocabulary ────────────────────────────────────────────────
// Loads vault/vocab.bin (length-prefixed byte strings, one per token).
// Returns true if at least one token loaded; vocab() then returns the
// table (sized vocab_size at load time — never indexed past it).
bool load_vocab(const char* path, uint32_t vocab_size, bool debug);
[[nodiscard]] const std::string* vocab() noexcept;   // nullptr until loaded

// ── Expert popularity histogram ───────────────────────────────
// Claim counts per (absolute layer, expert), persisted across runs so
// prewarm can stream the historically hottest experts first. `hits`
// is the caller-owned [128 × prefill::HITS_STRIDE] array.
void load_expert_freq(uint32_t* hits);
void save_expert_freq(const uint32_t* hits);

// ── Popularity-ordered cache prewarm ──────────────────────────
// Streams the historically hottest experts into the ring while the
// engine is idle: at startup and, in serve mode, between requests.
// set_busy(true) pauses it so it never competes with demand reads;
// rearm() re-arms a pass with the freshly updated histogram.
// stop() joins the thread — call before the streamer is destroyed.
class Prewarm {
public:
    void start(smoe::io::Streamer* streamer, const uint32_t* hits,
               uint32_t start_layer, uint32_t num_moe_layers,
               uint32_t experts_per_layer);
    void set_busy(bool busy) noexcept {
        busy_.store(busy, std::memory_order_relaxed);
    }
    void rearm() noexcept {
        epoch_.fetch_add(1, std::memory_order_relaxed);
    }
    void stop() noexcept;

private:
    std::atomic<bool>     shutdown_ { false };
    std::atomic<bool>     busy_     { false };
    std::atomic<uint32_t> epoch_    { 1 };
    std::thread           thread_;
};

// ── Pre-allocated engine buffers ──────────────────────────────
// Every activation plane the forward pass touches, allocated once at
// boot on 16 KB boundaries and registered with Metal where the GPU
// reads or writes them. The token loop never allocates.
struct Buffers {
    // Multi-layer KV ring: [layers][2][attn_ctx][kv_dim]
    float* kv_cache   { nullptr };

    // Decode-step planes (one token)
    float* hidden     { nullptr };   // [d_model] residual stream
    float* normed     { nullptr };   // [d_model] post-norm scratch
    float* qbuf       { nullptr };   // [q_dim]
    float* kbuf       { nullptr };   // [kv_dim]
    float* vbuf       { nullptr };   // [kv_dim]
    float* attn_out   { nullptr };   // [q_dim]
    float* routed_out { nullptr };   // [d_model] routed-expert accumulator
    float* shared_out { nullptr };   // [d_model] shared-expert output
    float* expert_out { nullptr };   // [MAX_ACTIVE × d_model] GPU FFN rows
    float* l0_gate    { nullptr };   // [ffn_dim] dense-L0 scratch
    float* l0_up      { nullptr };   // [ffn_dim]
    float* sh_gate    { nullptr };   // [shared_dim] shared-expert scratch
    float* sh_up      { nullptr };   // [shared_dim]
    float* gate_scores{ nullptr };   // [max_experts_per_layer] router logits

    // RoPE tables: inv_freq once per boot, cos/sin once per token step.
    float* rope_inv_freq { nullptr };  // [head_dim/2]
    float* rope_cos      { nullptr };  // [head_dim/2]
    float* rope_sin      { nullptr };  // [head_dim/2]

    // Layer-major prefill activation planes (CHUNK rows each).
    smoe::prefill::Buffers pf {};

    // Allocate + register everything. Aborts the process on OOM —
    // there is no meaningful recovery before the first token.
    void init(SmoeMetalCtx* metal, const SmoeModelConfig& cfg,
              const SmoeHeader& hdr, uint32_t attn_ctx);
};

} // namespace smoe::engine
