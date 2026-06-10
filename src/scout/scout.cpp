// ═══════════════════════════════════════════════════════════════
// scout.cpp — S-MoE Engine · Surface Scout (Week 4: Heuristic)
// ═══════════════════════════════════════════════════════════════
// Phase 4 — Week 4
//
// This is the bootstrap oracle.  Its job is NOT to be smart.
// Its job is to validate the full pipeline by producing expert
// predictions that are good enough to keep the ring buffer
// supplied, so that the GPU never stalls.
//
// Heuristic strategy (three layers):
//
//   ① N-gram recency bias:
//      The last NGRAM_WINDOW token IDs are hashed into a compact
//      key. We maintain a frequency table: key → expert histogram.
//      The most-activated experts for this n-gram context are
//      returned as the primary predictions.
//
//   ② Layer-wise recency bias:
//      If layer L activated expert E at step N, it is statistically
//      likely to activate experts in the neighbourhood [E-2, E+2]
//      at step N+1. This captures the "expert locality" property
//      observed in DeepSeek MoE ablation studies.
//
//   ③ Uniform fallback:
//      If neither heuristic produces enough candidates, we fill
//      the remaining slots with evenly-spaced expert IDs distributed
//      across the full expert count range.  This guarantees the
//      ring buffer is always fully seeded, even on a cold start.
//
// Week 5+: replace Impl with a distilled neural routing head.
//           The Scout interface (scout.hpp) remains unchanged.
//
// Design invariants:
//   ① No heap allocations inside forward() — all tables are
//     pre-allocated in the constructor.
//   ② All mutable state is confined to Impl — Scout is the thin
//     PIMPL wrapper.
//   ③ next_token_id in heuristic mode is produced by a simple
//     unigram frequency table over observed token history.
//     This is a deliberate placeholder — in Week 5+ the Scout
//     runs an actual language model forward pass.
// ═══════════════════════════════════════════════════════════════

#include "scout.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace smoe::scout {

// ── Heuristic constants ───────────────────────────────────────

// N-gram window: how many recent token IDs form the context key.
inline constexpr uint32_t NGRAM_WINDOW   = 4;
// Number of entries in the n-gram frequency table.
// Must be a power of two for the bitmask hash.
inline constexpr uint32_t NGRAM_TABLE    = 4096;
// Expert histogram: for each n-gram slot, track the top N_HIST
// most-recently-seen expert IDs per layer (approximate).
inline constexpr uint32_t N_HIST        = MAX_ACTIVE;

// Expert universe — must be set to ≥ actual experts per layer.
// DeepSeek-V2/V3 uses 256 routed experts per MoE layer.
inline constexpr uint32_t EXPERT_UNIVERSE = 256;

// Number of MoE layers we track recency for.
// DeepSeek-V2: 21 MoE layers.  DeepSeek-V3: 27+.  We cap at 64.
inline constexpr uint32_t MAX_LAYERS    = 64;

// Recency neighbourhood half-width for layer bias.
inline constexpr uint32_t RECENCY_DELTA = 3;

// Context history ring for next-token unigram prediction.
inline constexpr uint32_t HISTORY_CAP  = 1024;

// ── Data structures ────────────────────────────────────────────

// One n-gram bucket: stores the MAX_ACTIVE most-activated
// expert IDs for (layer_id, n-gram key) combinations.
// Using a simple circular eviction — no sorting needed.
struct NgramBucket {
    // Per-layer expert hint ring: [layer][slot] = expert_id
    uint32_t experts[MAX_LAYERS][N_HIST] {};
    uint32_t counts [MAX_LAYERS][N_HIST] {};
    uint32_t head   [MAX_LAYERS]         {};  // write head per layer
};

// Per-layer recency state: the expert IDs activated at the
// most recent step.
struct LayerRecency {
    uint32_t last_experts[MAX_ACTIVE] {};
    uint32_t count { 0 };
};

// ── Scout::Impl ───────────────────────────────────────────────
struct Scout::Impl {
    // N-gram table (pre-allocated, never resized)
    NgramBucket ngram_table[NGRAM_TABLE] {};

    // Per-layer recency state
    LayerRecency recency[MAX_LAYERS] {};

    // Rolling context window (ring buffer of last NGRAM_WINDOW tokens)
    uint32_t context_ring[NGRAM_WINDOW] {};
    uint32_t context_pos { 0 };
    uint32_t context_fill { 0 };  // how many tokens seen so far

    // Token history for simple frequency prediction
    uint32_t history[HISTORY_CAP] {};
    uint32_t history_pos  { 0 };
    uint32_t history_fill { 0 };

    // Token frequency table for next-token heuristic
    // Tracks counts for token IDs 0..FREQ_TABLE_SIZE-1.
    static constexpr uint32_t FREQ_TABLE_SIZE = 65536;
    uint32_t token_freq[FREQ_TABLE_SIZE] {};

    // Configuration loaded from the .smoe vault (or defaults)
    uint32_t num_moe_layers   { 21 };   // DeepSeek-V2 default
    uint32_t experts_per_layer{ EXPERT_UNIVERSE };

    // Volatile step counter (monotonic)
    uint64_t step { 0 };

    // ── Helpers ─────────────────────────────────────────────

    // Hash the current context ring into an n-gram table index.
    uint32_t ngram_hash() const noexcept {
        // FNV-1a over the context ring tokens
        uint32_t h = 2166136261u;
        for (uint32_t i = 0; i < NGRAM_WINDOW; ++i) {
            uint32_t tok = context_ring[(context_pos + i) % NGRAM_WINDOW];
            h ^= tok;
            h *= 16777619u;
        }
        return h & (NGRAM_TABLE - 1);
    }

    // Record that (layer_id, expert_id) was observed at this step.
    void observe(uint32_t layer_id, uint32_t expert_id) noexcept {
        if (layer_id >= MAX_LAYERS) return;
        NgramBucket& bucket = ngram_table[ngram_hash()];
        uint32_t     slot   = bucket.head[layer_id] % N_HIST;
        bucket.experts[layer_id][slot] = expert_id;
        bucket.counts [layer_id][slot]++;
        bucket.head   [layer_id] = slot + 1;

        // Update recency
        LayerRecency& rec = recency[layer_id];
        if (rec.count < MAX_ACTIVE) {
            rec.last_experts[rec.count++] = expert_id;
        } else {
            // Circular eviction
            rec.last_experts[step % MAX_ACTIVE] = expert_id;
        }
    }

    // Produce a prediction for (layer_id) K steps ahead.
    // Fills out[] with up to MAX_ACTIVE expert IDs. Returns count.
    uint32_t predict(uint32_t layer_id, uint32_t* out) const noexcept {
        if (layer_id >= MAX_LAYERS) return 0;

        uint32_t n = 0;
        bool     seen[EXPERT_UNIVERSE] {};  // dedup mask — stack-allocated

        // ── Strategy 1: n-gram bucket ───────────────────────
        const NgramBucket& bucket = ngram_table[ngram_hash()];
        for (uint32_t i = 0; i < N_HIST && n < MAX_ACTIVE; ++i) {
            uint32_t e = bucket.experts[layer_id][i];
            if (e < experts_per_layer && !seen[e]) {
                out[n++] = e;
                seen[e]  = true;
            }
        }

        // ── Strategy 2: recency neighbourhood ───────────────
        const LayerRecency& rec = recency[layer_id];
        for (uint32_t r = 0; r < rec.count && n < MAX_ACTIVE; ++r) {
            uint32_t center = rec.last_experts[r];
            // Walk the neighbourhood [center - DELTA, center + DELTA]
            for (int32_t d = -int32_t(RECENCY_DELTA);
                 d <= int32_t(RECENCY_DELTA) && n < MAX_ACTIVE; ++d)
            {
                int32_t e = int32_t(center) + d;
                if (e < 0 || uint32_t(e) >= experts_per_layer) continue;
                if (!seen[uint32_t(e)]) {
                    out[n++]          = uint32_t(e);
                    seen[uint32_t(e)] = true;
                }
            }
        }

        // ── Strategy 3: uniform fallback ────────────────────
        // Distribute evenly across [0, experts_per_layer)
        if (n < MAX_ACTIVE && experts_per_layer > 0) {
            uint32_t step_size = experts_per_layer / MAX_ACTIVE;
            if (step_size == 0) step_size = 1;
            for (uint32_t i = 0; i < experts_per_layer && n < MAX_ACTIVE; i += step_size) {
                if (!seen[i]) {
                    out[n++] = i;
                    seen[i]  = true;
                }
            }
        }

        return n;
    }

    // Heuristic next-token: return the most frequent recent token
    // that isn't the current one (simple bigram).
    uint32_t next_token_heuristic(uint32_t current_token) const noexcept {
        // Look for the most frequent token in history after `current_token`
        uint32_t best_tok   = (current_token + 1) % FREQ_TABLE_SIZE;
        uint32_t best_count = 0;

        // Scan the last NGRAM_WINDOW history entries for bigrams
        for (uint32_t i = 0; i + 1 < std::min(history_fill, HISTORY_CAP); ++i) {
            uint32_t h_idx = (history_pos + i) % HISTORY_CAP;
            uint32_t n_idx = (h_idx + 1) % HISTORY_CAP;
            if (history[h_idx] == current_token) {
                uint32_t candidate = history[n_idx];
                if (candidate < FREQ_TABLE_SIZE &&
                    token_freq[candidate] > best_count)
                {
                    best_count = token_freq[candidate];
                    best_tok   = candidate;
                }
            }
        }
        return best_tok;
    }

    // Push a new token into the context window and frequency table.
    void push_token(uint32_t token_id) noexcept {
        context_ring[context_pos % NGRAM_WINDOW] = token_id;
        context_pos  = (context_pos + 1) % NGRAM_WINDOW;
        context_fill = std::min(context_fill + 1, NGRAM_WINDOW);

        history[history_pos] = token_id;
        history_pos          = (history_pos + 1) % HISTORY_CAP;
        history_fill         = std::min(history_fill + 1, HISTORY_CAP);

        if (token_id < FREQ_TABLE_SIZE) {
            token_freq[token_id]++;
        }
    }
};

// ── Scout public API ──────────────────────────────────────────

Scout::Scout(const char* scout_safetensors_path)
    : impl_(new Impl())
{
    // In Week 5+, this will:
    //   1. Open the .safetensors file
    //   2. Parse the tensor index
    //   3. mmap the weight blobs into Unified Memory
    //   4. Initialise the neural routing head
    //
    // For the Week 4 heuristic Scout, we just log the path and
    // proceed with zeroed state tables (cold start).
    if (scout_safetensors_path) {
        std::fprintf(stderr,
            "[scout] Heuristic mode — ignoring weights at: %s\n"
            "        (Week 4: statistical oracle; Week 5+ loads real weights)\n",
            scout_safetensors_path);
    } else {
        std::fprintf(stderr,
            "[scout] Heuristic mode — no weight path provided.\n");
    }
}

Scout::~Scout() {
    delete impl_;
}

ScoutOutput Scout::forward(uint32_t token_id) {
    ScoutOutput out {};

    // Push token into context
    impl_->push_token(token_id);

    // Predict next token (heuristic bigram)
    out.next_token_id = impl_->next_token_heuristic(token_id);

    // Fill K-step lookahead for all MoE layers
    //
    // Heuristic: the expert activation pattern is assumed to be
    // similar across future steps (recency dominates at K=1..3,
    // uniform fallback dominates at K=7..10).
    // Each lookahead step is offset by one "recency shift" to
    // model the slight drift in expert activations over time.
    for (uint32_t k = 0; k < LOOKAHEAD_K; ++k) {
        ExpertPrediction& pred = out.lookahead[k];
        // We predict for one representative MoE layer per lookahead
        // step, cycling through the known MoE layers.
        // The main loop will call prefetch() for each prediction.
        pred.layer_id = k % impl_->num_moe_layers;
        pred.count    = impl_->predict(pred.layer_id, pred.expert_ids);
    }

    impl_->step++;
    return out;
}

void Scout::reset_context() {
    // Zero all context state but preserve the learned frequency tables —
    // those carry information that is useful across prompts.
    std::memset(impl_->context_ring, 0, sizeof(impl_->context_ring));
    impl_->context_pos  = 0;
    impl_->context_fill = 0;

    std::memset(impl_->history, 0, sizeof(impl_->history));
    impl_->history_pos  = 0;
    impl_->history_fill = 0;

    // Reset per-layer recency
    for (auto& rec : impl_->recency) {
        rec = LayerRecency{};
    }

    impl_->step = 0;
    std::fprintf(stderr, "[scout] Context reset.\n");
}

} // namespace smoe::scout
