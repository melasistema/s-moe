// ═══════════════════════════════════════════════════════════════
// scout.hpp — S-MoE Engine · Surface Scout Interface
// ═══════════════════════════════════════════════════════════════
// The Surface Scout is the heavy model's own dense backbone
// (~16 GB bf16 for Qwen3-235B): embeddings, attention projections,
// norms, router gates, and the LM head — everything the sculptor
// did NOT exile into the .smoe expert vault.
//
// It is a weights artifact, not a running model. The engine's
// forward pass (main.cpp / prefill.cpp) reads these tensors
// directly via the accessors below; routing is computed exactly
// from the heavy hidden state with get_gate() + compute_top_k().
//
// History: this class once ran its own speculative forward pass
// to predict expert routing K steps ahead. Measurement retired it
// (51.5% oracle accuracy vs 46.4% free ring retention) — the
// forward machinery, its KV mirror (~1.6 GB), and the Week-4
// heuristic oracle were deleted, not just bypassed.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "../common.hpp"

#include <cstdint>

struct SmoeMetalCtx;

namespace smoe::scout {

// Maximum simultaneously active experts per token per layer
inline constexpr uint32_t MAX_ACTIVE    =  16;

// ── Expert selection for one token step at one layer ─────────
struct ExpertPrediction {
    uint32_t layer_id;
    uint32_t expert_ids[MAX_ACTIVE];
    float    expert_weights[MAX_ACTIVE];
    uint32_t count;   // actual active count ≤ MAX_ACTIVE
};

inline constexpr uint32_t MAX_MOE_LAYERS = 128;

// ── Surface Scout — resident dense-backbone weights ──────────
class Scout {
public:
    // Load Scout weights from a .safetensors file (mmap, resident).
    Scout(const char* scout_safetensors_path, SmoeMetalCtx* metal_ctx, const SmoeHeader* vault_hdr = nullptr);
    ~Scout();

    // Get the dynamically parsed model configuration
    const SmoeModelConfig& config() const noexcept;

    // ── Accessors (for heavy-model / debugger) ────────────────────────
    const uint16_t* get_embed() const noexcept;
    const uint16_t* get_lm_head() const noexcept;
    const uint16_t* get_model_norm() const noexcept;

    // Layer 0: dense MLP
    const uint16_t* get_l0_gate() const noexcept;
    const uint16_t* get_l0_up() const noexcept;
    const uint16_t* get_l0_down() const noexcept;

    // Layers 0..27
    const uint16_t* get_q_proj(uint32_t layer) const noexcept;
    const uint16_t* get_k_proj(uint32_t layer) const noexcept;
    const uint16_t* get_v_proj(uint32_t layer) const noexcept;
    const uint16_t* get_q_norm(uint32_t layer) const noexcept;
    const uint16_t* get_k_norm(uint32_t layer) const noexcept;
    const uint16_t* get_o_proj(uint32_t layer) const noexcept;
    uint32_t compute_top_k(const float* scores, uint32_t n, uint32_t k, uint32_t* out_indices, float* out_weights, bool norm_topk) const noexcept;
    const uint16_t* get_input_norm(uint32_t layer) const noexcept;
    [[nodiscard]] const uint16_t* get_post_norm(uint32_t l) const noexcept;

    // Mmapped memory access
    [[nodiscard]] const void* get_mapped_ptr() const noexcept;
    [[nodiscard]] size_t      get_mapped_size() const noexcept;
    const uint16_t* get_gate(uint32_t layer) const noexcept;

    float* get_lm_head_scores() const noexcept;

    // Layers 1..27 shared experts
    const uint16_t* get_shared_gate(uint32_t layer) const noexcept;
    const uint16_t* get_shared_up(uint32_t layer) const noexcept;
    const uint16_t* get_shared_down(uint32_t layer) const noexcept;

    Scout(const Scout&)            = delete;
    Scout& operator=(const Scout&) = delete;

private:
    struct Impl;
    Impl* impl_ { nullptr };
};

} // namespace smoe::scout
