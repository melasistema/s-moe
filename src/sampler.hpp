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

// Pre-allocated top-K candidate buffer. The one allocation happens here,
// at startup — sample_token itself is heap-free (Golden Rule: no dynamic
// memory in the token loop). Sized to the effective top-k: the top-p/top-k
// walk never looks past the first top_k probability-sorted tokens, so the
// full vocab never needs to be materialised or sorted.
struct SamplerScratch {
    struct ProbTok { float p; uint32_t v; };
    std::vector<ProbTok> topk;

    SamplerScratch(uint32_t top_k, uint32_t vocab_size) {
        topk.resize(std::min(std::max(top_k, 1u), vocab_size));
    }
};

static inline uint32_t sample_token(
    float* scores,
    const SamplerConfig& cfg,
    const uint32_t* token_history,
    uint32_t history_len,
    std::mt19937& rng,
    SamplerScratch& scratch)
{
    // ── Repetition penalty ──
    if (cfg.rep_penalty != 1.0f) {
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
            // Penalise each distinct token once. Duplicate check is a scan
            // of the ≤256-token window (≤32K compares) — cheaper than the
            // old vocab-sized std::vector<bool> allocated every call.
            bool seen = false;
            for (uint32_t j = start_idx; j < i; ++j) {
                if (token_history[j] == tok) { seen = true; break; }
            }
            if (seen) continue;
            if (scores[tok] <= 0) {
                scores[tok] *= cfg.rep_penalty;
            } else {
                scores[tok] /= cfg.rep_penalty;
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

    // Single pass over the vocab: accumulate the full softmax denominator
    // while a fixed-size min-heap (root = smallest kept candidate) retains
    // the K largest unnormalised probabilities. Replaces the old
    // full-vocab sort of two 152K-element heap-allocated vectors per token.
    SamplerScratch::ProbTok* heap = scratch.topk.data();
    const uint32_t K = static_cast<uint32_t>(scratch.topk.size());

    auto sift_down = [heap](uint32_t i, uint32_t n) {
        for (;;) {
            uint32_t l = 2 * i + 1;
            if (l >= n) break;
            uint32_t m = (heap[l].p < heap[i].p) ? l : i;
            uint32_t r = l + 1;
            if (r < n && heap[r].p < heap[m].p) m = r;
            if (m == i) break;
            SamplerScratch::ProbTok tmp = heap[i];
            heap[i] = heap[m];
            heap[m] = tmp;
            i = m;
        }
    };

    float    sum_exp  = 0.0f;
    uint32_t heap_len = 0;
    for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
        float e = std::exp((scores[v] - max_score) / cfg.temperature);
        sum_exp += e;
        if (heap_len < K) {
            heap[heap_len].p = e;
            heap[heap_len].v = v;
            if (++heap_len == K) {
                for (uint32_t i = K / 2; i-- > 0; ) sift_down(i, K);
            }
        } else if (e > heap[0].p) {
            heap[0].p = e;
            heap[0].v = v;
            sift_down(0, K);
        }
    }

    // Descending-probability order for the top-p / top-k cutoff walk.
    std::sort(heap, heap + heap_len,
              [](const SamplerScratch::ProbTok& a, const SamplerScratch::ProbTok& b) {
        return a.p > b.p;
    });

    float    inv_sum = 1.0f / sum_exp;
    float    cumsum  = 0.0f;
    uint32_t top_len = 0;
    for (uint32_t i = 0; i < heap_len; ++i) {
        heap[i].p *= inv_sum;
        cumsum += heap[i].p;
        top_len++;
        if (cumsum >= cfg.top_p || top_len >= cfg.top_k) break;
    }

    float rand_val = std::generate_canonical<float, 24>(rng) * cumsum;
    float running_sum = 0.0f;
    for (uint32_t i = 0; i < top_len; ++i) {
        running_sum += heap[i].p;
        if (rand_val <= running_sum) {
            return heap[i].v;
        }
    }

    return heap[top_len - 1].v;
}

} // namespace smoe
