// ═══════════════════════════════════════════════════════════════
// scout.hpp — S-MoE Engine · Surface Scout Interface
// ═══════════════════════════════════════════════════════════════
// Phase 3/4 stub — to be implemented in Week 4+.
//
// The Surface Scout is a lightweight (~1.5B param) model that:
//   ① Performs standard autoregressive token generation.
//   ② Runs a secondary routing head that predicts which expert
//      IDs will be activated up to K steps into the future.
//
// Contract for the core loop (main.cpp):
//   scout.forward(context) →
//     { next_token_id, predicted_expert_ids[K][MAX_ACTIVE] }
//
// Week 4 (heuristic): expert prediction via n-gram attention maps.
// Week 5+:            replace with distilled neural routing head.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "../common.hpp"

#include <cstdint>
#include <span>
#include <vector>

struct SmoeMetalCtx;

namespace smoe::scout {

// Lookahead window depth — how many future steps to predict
inline constexpr uint32_t LOOKAHEAD_K   = 10;
// Maximum simultaneously active experts per token per layer
inline constexpr uint32_t MAX_ACTIVE    =  6;

// ── Expert prediction for one future step ────────────────────
struct ExpertPrediction {
    uint32_t layer_id;
    uint32_t expert_ids[MAX_ACTIVE];
    float    expert_weights[MAX_ACTIVE];
    uint32_t count;   // actual active count ≤ MAX_ACTIVE
};

inline constexpr uint32_t MAX_MOE_LAYERS = 128;

// ── Scout forward-pass result ────────────────────────────────
struct ScoutOutput {
    uint32_t next_token_id;
    // Predicted experts for the current token for all MoE layers
    ExpertPrediction routing[MAX_MOE_LAYERS];
};

// ── Surface Scout — Week 4 interface stub ────────────────────
class Scout {
public:
    // Load Scout weights from a .safetensors file.
    Scout(const char* scout_safetensors_path, SmoeMetalCtx* metal_ctx);
    ~Scout();

    // Run one forward step, updating internal KV-cache context.
    // Returns next token + expert routing for the current token.
    ScoutOutput forward(uint32_t token_id);

    // Rollback the internal KV-cache state by K steps to recover from 
    // speculative divergence.
    void rollback(uint32_t steps);

    // Reset KV-cache for a new prompt.
    void reset_context();

    // Write key/value directly to the Scout's KV cache to keep it in sync with Heavy model.
    void write_kv_cache(uint32_t layer, uint32_t slot, const float* k, const float* v);

    // Get the dynamically parsed model configuration
    const SmoeModelConfig& config() const noexcept;

    // ── Getters for full-model execution ────────────────────────
    const float* get_embed() const noexcept;
    const float* get_lm_head() const noexcept;
    const float* get_model_norm() const noexcept;
    
    // Layer 0: dense MLP
    const float* get_l0_gate() const noexcept;
    const float* get_l0_up() const noexcept;
    const float* get_l0_down() const noexcept;

    // Layers 0..27
    const float* get_q_proj(uint32_t layer) const noexcept;
    const float* get_k_proj(uint32_t layer) const noexcept;
    const float* get_v_proj(uint32_t layer) const noexcept;
    const float* get_o_proj(uint32_t layer) const noexcept;
    const float* get_input_norm(uint32_t layer) const noexcept;
    const float* get_post_norm(uint32_t layer) const noexcept;
    
    float* get_lm_head_scores() const noexcept;
    
    // Layers 1..27 shared experts
    const float* get_shared_gate(uint32_t layer) const noexcept;
    const float* get_shared_up(uint32_t layer) const noexcept;
    const float* get_shared_down(uint32_t layer) const noexcept;

    Scout(const Scout&)            = delete;
    Scout& operator=(const Scout&) = delete;

private:
    struct Impl;
    Impl* impl_ { nullptr };
};

} // namespace smoe::scout
