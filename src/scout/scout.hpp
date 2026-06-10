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
inline constexpr uint32_t MAX_ACTIVE    =  8;

// ── Expert prediction for one future step ────────────────────
struct ExpertPrediction {
    uint32_t layer_id;
    uint32_t expert_ids[MAX_ACTIVE];
    uint32_t count;   // actual active count ≤ MAX_ACTIVE
};

// ── Scout forward-pass result ────────────────────────────────
struct ScoutOutput {
    uint32_t next_token_id;
    // Predicted experts for steps [N+1 … N+K]
    ExpertPrediction lookahead[LOOKAHEAD_K];
};

// ── Surface Scout — Week 4 interface stub ────────────────────
class Scout {
public:
    // Load Scout weights from a .safetensors file.
    Scout(const char* scout_safetensors_path, SmoeMetalCtx* metal_ctx);
    ~Scout();

    // Run one forward step, updating internal KV-cache context.
    // Returns next token + K-step expert lookahead.
    ScoutOutput forward(uint32_t token_id);

    // Reset KV-cache for a new prompt.
    void reset_context();

    Scout(const Scout&)            = delete;
    Scout& operator=(const Scout&) = delete;

private:
    struct Impl;
    Impl* impl_ { nullptr };
};

} // namespace smoe::scout
