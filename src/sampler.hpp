#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace smoe {

struct SamplerConfig {
    uint32_t vocab_size;
    float    temperature;
    float    top_p;
    uint32_t top_k;
    float    rep_penalty;
};

static inline uint32_t sample_token(
    float* scores, 
    const SamplerConfig& cfg, 
    const uint32_t* token_history, 
    uint32_t history_len,
    std::mt19937& rng) 
{
    // ── Repetition penalty ──
    if (cfg.rep_penalty != 1.0f) {
        std::vector<bool> penalized(cfg.vocab_size, false);
        uint32_t penalty_last_n = 256; // 256 tokens is enough to break loops without forcing language switches
        uint32_t start_idx = (history_len > penalty_last_n) ? (history_len - penalty_last_n) : 0;
        
        for (uint32_t i = start_idx; i < history_len; ++i) {
            uint32_t tok = token_history[i];
            // Exempt common punctuation and special tokens from penalty.
            // NOTE: these IDs are vocab-specific. Qwen3 byte-level BPE:
            // 13: '.', 11: ',', 198: '\n', 271: '\n\n', 30: '?', 0: '!',
            // 151643: <|endoftext|>, 151645: <|im_end|>
            if (tok == 13 || tok == 11 || tok == 198 || tok == 271 || tok == 30 || tok == 0 ||
                tok == 151643 || tok == 151645) {
                continue;
            }
            if (!penalized[tok]) {
                if (scores[tok] <= 0) {
                    scores[tok] *= cfg.rep_penalty;
                } else {
                    scores[tok] /= cfg.rep_penalty;
                }
                penalized[tok] = true;
            }
        }
    }

    // ── Greedy ──
    if (cfg.temperature < 1e-4f) {
        float best_score = -1e38f;
        uint32_t best_tok = 0;
        for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
            if (scores[v] > best_score) {
                best_score = scores[v];
                best_tok   = v;
            }
        }
        return best_tok;
    }

    // ── Temperature + Top-P + Top-K ──
    float max_score = scores[0];
    for (uint32_t v = 1; v < cfg.vocab_size; ++v) {
        if (scores[v] > max_score) max_score = scores[v];
    }
    
    float sum_exp = 0.0f;
    struct ProbTok { float p; uint32_t v; };
    std::vector<ProbTok> probs(cfg.vocab_size);
    for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
        probs[v].v = v;
        probs[v].p = std::exp((scores[v] - max_score) / cfg.temperature);
        sum_exp += probs[v].p;
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
        probs[v].p *= inv_sum;
    }
    
    std::sort(probs.begin(), probs.end(), [](const ProbTok& a, const ProbTok& b) {
        return a.p > b.p;
    });
    
    float cumsum = 0.0f;
    uint32_t top_k_len = 0;
    for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
        cumsum += probs[v].p;
        top_k_len++;
        if (cumsum >= cfg.top_p || top_k_len >= cfg.top_k) break;
    }
    
    float rand_val = std::generate_canonical<float, 24>(rng) * cumsum;
    float running_sum = 0.0f;
    for (uint32_t v = 0; v < top_k_len; ++v) {
        running_sum += probs[v].p;
        if (rand_val <= running_sum) {
            return probs[v].v;
        }
    }
    
    return probs[top_k_len - 1].v;
}

} // namespace smoe
