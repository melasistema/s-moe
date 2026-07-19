// ═══════════════════════════════════════════════════════════════
// prefill.cpp — S-MoE Engine · Layer-Major Batched Prefill
// ═══════════════════════════════════════════════════════════════
// See prefill.hpp for the design rationale. Structure per chunk:
//
//   embed all B tokens → for each layer L:
//     ① RMS-norm (CPU), then ALL B tokens' Q/K/V in one command buffer
//     ② QK-norm + RoPE + K/V ring append per token (CPU)
//     ③ causal attention per token (window = its own history only)
//     ④ ALL B o_proj rows in one command buffer, then residual +
//        FFN norm per token (CPU)
//     ⑤ router gate per token → per-expert token lists (dedupe);
//        every FRESH expert fires at the streamer immediately, so the
//        NVMe works while later tokens are still routing
//     ⑥ claim each union expert ONCE and run the fused FFN for every
//        token routed to it
// ═══════════════════════════════════════════════════════════════

#include "prefill.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <thread>

namespace smoe::prefill {

namespace {

// Per-expert token list for one (layer, chunk): which chunk rows routed
// to this expert and with what gate weight. Static: single-threaded
// caller, zero allocation in the hot path (CLAUDE.md).
inline constexpr uint32_t MAX_EXPERTS = 512;
// Routing width comes from cfg.moe_top_k (vault arch block, 8 for Qwen3);
// buffers are sized for the engine-wide ceiling.
inline constexpr uint32_t TOPK_MAX = smoe::scout::MAX_ACTIVE;

struct ExpertTokens {
    uint8_t tok[CHUNK];
    float   w[CHUNK];
    uint32_t count;
};
ExpertTokens g_elist[MAX_EXPERTS];
uint32_t     g_union_ids[MAX_EXPERTS];
uint32_t     g_union_size;

void union_reset() { g_union_size = 0; }

// Returns true the first time this expert enters the layer's union —
// the caller fires the streamer on that edge, so I/O for the layer
// starts while later tokens are still routing.
bool union_add(uint32_t expert, uint32_t tok_row, float weight) {
    ExpertTokens& et = g_elist[expert];
    bool fresh = true;
    for (uint32_t u = 0; u < g_union_size; ++u) {
        if (g_union_ids[u] == expert) { fresh = false; break; }
    }
    if (fresh) {
        et.count = 0;
        if (g_union_size < MAX_EXPERTS) g_union_ids[g_union_size++] = expert;
    }
    if (et.count < CHUNK) {
        et.tok[et.count] = static_cast<uint8_t>(tok_row);
        et.w[et.count]   = weight;
        et.count++;
    }
    return fresh;
}

// Batched dense-dispatch pointer tables (QKV: 3 entries per token;
// o_proj: 1). Static: single-threaded caller, zero allocation in the
// hot path (CLAUDE.md).
inline constexpr uint32_t MAX_DISPATCH = 3 * CHUNK;
const uint16_t* g_disp_w[MAX_DISPATCH];
const float*    g_disp_in[MAX_DISPATCH];
float*          g_disp_out[MAX_DISPATCH];
uint32_t        g_disp_rows[MAX_DISPATCH];
uint32_t        g_disp_cols[MAX_DISPATCH];

} // namespace

void run_layer_major(const Params& P,
                     const uint32_t* tokens, uint32_t count,
                     uint64_t stream_base,
                     uint32_t& ctx_pos, uint32_t& ctx_fill,
                     float* attn_scores)
{
    const SmoeModelConfig& cfg = *P.cfg;
    const ExpertLayout& lay = *P.layout;
    smoe::scout::Scout& scout = *P.scout;
    smoe::io::Streamer& streamer = *P.streamer;
    const Buffers& b = P.b;

    const uint32_t d_model   = cfg.d_model;
    const uint32_t q_dim     = cfg.num_heads * cfg.head_dim;
    const uint32_t kv_dim    = cfg.num_kv_heads * cfg.head_dim;
    const uint32_t rope_half = cfg.head_dim / 2;
    const uint32_t ATTN_CTX  = P.attn_ctx;
    const uint32_t num_layers = cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0);
    const uint32_t n_experts = cfg.max_experts_per_layer;
    const uint32_t heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
    const float    attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    for (uint32_t chunk = 0; chunk < count; chunk += CHUNK) {
        const uint32_t B = (count - chunk < CHUNK) ? (count - chunk) : CHUNK;
        const uint64_t base_pos  = stream_base + chunk;
        const uint32_t base_slot = ctx_pos;
        const uint32_t base_fill = ctx_fill;

        // ── Embeddings + per-token RoPE tables ────────────────
        for (uint32_t i = 0; i < B; ++i) {
            const uint16_t* emb = scout.get_embed()
                + static_cast<size_t>(tokens[chunk + i]) * d_model;
            float* h = b.hidden + static_cast<size_t>(i) * d_model;
            for (uint32_t d = 0; d < d_model; ++d) h[d] = smoe::bf16_to_f32(emb[d]);

            float* rc = b.rope_cos + static_cast<size_t>(i) * rope_half;
            float* rs = b.rope_sin + static_cast<size_t>(i) * rope_half;
            for (uint32_t d = 0; d < rope_half; ++d) {
                float angle = static_cast<float>(base_pos + i) * P.rope_inv_freq[d];
                rc[d] = std::cos(angle);
                rs[d] = std::sin(angle);
            }
        }

        for (uint32_t l = 0; l < num_layers; ++l) {
            float* k_cache = P.full_kv_cache + (static_cast<size_t>(l) * 2 * ATTN_CTX + 0 * ATTN_CTX) * kv_dim;
            float* v_cache = P.full_kv_cache + (static_cast<size_t>(l) * 2 * ATTN_CTX + 1 * ATTN_CTX) * kv_dim;

            // ── ① Input norm (CPU), then ALL B tokens' Q/K/V in one
            //     command buffer. The per-token synchronous matvec
            //     round-trips (2×B×layers command buffers, each
            //     re-reading the full weight matrices at thread-per-row
            //     bandwidth) were the prefill's dominant non-I/O cost.
            for (uint32_t i = 0; i < B; ++i) {
                float* h  = b.hidden + static_cast<size_t>(i) * d_model;
                float* nr = b.normed + static_cast<size_t>(i) * d_model;
                std::memcpy(nr, h, d_model * sizeof(float));
                smoe::rms_norm_bf16(nr, scout.get_input_norm(l), d_model);
            }
            if (P.metal) {
                uint32_t n = 0;
                for (uint32_t i = 0; i < B; ++i) {
                    const float* nr = b.normed + static_cast<size_t>(i) * d_model;
                    const uint16_t* w[3] = { scout.get_q_proj(l), scout.get_k_proj(l), scout.get_v_proj(l) };
                    float* o[3] = { b.qbuf + static_cast<size_t>(i) * q_dim,
                                    b.kbuf + static_cast<size_t>(i) * kv_dim,
                                    b.vbuf + static_cast<size_t>(i) * kv_dim };
                    const uint32_t r[3] = { q_dim, kv_dim, kv_dim };
                    for (uint32_t j = 0; j < 3; ++j) {
                        g_disp_w[n] = w[j];  g_disp_in[n] = nr;  g_disp_out[n] = o[j];
                        g_disp_rows[n] = r[j];  g_disp_cols[n] = d_model;  ++n;
                    }
                }
                smoe_metal_scout_matvec_group_bf16(P.metal, g_disp_w, g_disp_in,
                                                   g_disp_out, g_disp_rows, g_disp_cols, n);
            } else {
                for (uint32_t i = 0; i < B; ++i) {
                    const float* nr = b.normed + static_cast<size_t>(i) * d_model;
                    smoe::matvec_bf16(b.qbuf + static_cast<size_t>(i) * q_dim,  scout.get_q_proj(l), nr, q_dim, d_model);
                    smoe::matvec_bf16(b.kbuf + static_cast<size_t>(i) * kv_dim, scout.get_k_proj(l), nr, kv_dim, d_model);
                    smoe::matvec_bf16(b.vbuf + static_cast<size_t>(i) * kv_dim, scout.get_v_proj(l), nr, kv_dim, d_model);
                }
            }

            // ── ② QK-norm + RoPE + K/V append per token (CPU) ──
            for (uint32_t i = 0; i < B; ++i) {
                float* qi  = b.qbuf + static_cast<size_t>(i) * q_dim;
                float* ki  = b.kbuf + static_cast<size_t>(i) * kv_dim;
                float* vi  = b.vbuf + static_cast<size_t>(i) * kv_dim;

                const uint16_t* q_norm_w = scout.get_q_norm(l);
                const uint16_t* k_norm_w = scout.get_k_norm(l);
                if (q_norm_w) {
                    for (uint32_t hh = 0; hh < cfg.num_heads; ++hh) {
                        smoe::rms_norm_bf16(qi + hh * cfg.head_dim, q_norm_w, cfg.head_dim);
                    }
                }
                if (k_norm_w) {
                    for (uint32_t hh = 0; hh < cfg.num_kv_heads; ++hh) {
                        smoe::rms_norm_bf16(ki + hh * cfg.head_dim, k_norm_w, cfg.head_dim);
                    }
                }

                const float* rc = b.rope_cos + static_cast<size_t>(i) * rope_half;
                const float* rs = b.rope_sin + static_cast<size_t>(i) * rope_half;
                for (uint32_t hh = 0; hh < cfg.num_heads; ++hh) {
                    for (uint32_t d = 0; d < rope_half; ++d) {
                        float q0 = qi[hh * cfg.head_dim + d];
                        float q1 = qi[hh * cfg.head_dim + d + rope_half];
                        qi[hh * cfg.head_dim + d]             = q0 * rc[d] - q1 * rs[d];
                        qi[hh * cfg.head_dim + d + rope_half] = q0 * rs[d] + q1 * rc[d];
                    }
                }
                for (uint32_t hh = 0; hh < cfg.num_kv_heads; ++hh) {
                    for (uint32_t d = 0; d < rope_half; ++d) {
                        float k0 = ki[hh * cfg.head_dim + d];
                        float k1 = ki[hh * cfg.head_dim + d + rope_half];
                        ki[hh * cfg.head_dim + d]             = k0 * rc[d] - k1 * rs[d];
                        ki[hh * cfg.head_dim + d + rope_half] = k0 * rs[d] + k1 * rc[d];
                    }
                }

                const uint32_t slot = (base_slot + i) % ATTN_CTX;
                std::memcpy(k_cache + static_cast<size_t>(slot) * kv_dim, ki, kv_dim * sizeof(float));
                std::memcpy(v_cache + static_cast<size_t>(slot) * kv_dim, vi, kv_dim * sizeof(float));
            }

            // ── ③ Causal attention per token (CPU) ──
            // All chunk K/V rows are already in the ring; causality holds
            // because token i's window is capped at its own history.
            for (uint32_t i = 0; i < B; ++i) {
                float* qi = b.qbuf + static_cast<size_t>(i) * q_dim;
                float* ao = b.attn_out + static_cast<size_t>(i) * q_dim;

                const uint32_t slot_i  = (base_slot + i) % ATTN_CTX;
                const uint32_t hist    = base_fill + i;
                const uint32_t valid   = (hist < ATTN_CTX) ? hist + 1 : ATTN_CTX;
                std::memset(ao, 0, q_dim * sizeof(float));

                for (uint32_t hh = 0; hh < cfg.num_heads; ++hh) {
                    uint32_t kv_h = hh / heads_per_kv;
                    const float* qhead = qi + hh * cfg.head_dim;
                    for (uint32_t j = 0; j < valid; ++j) {
                        uint32_t kslot = (slot_i - j + ATTN_CTX) % ATTN_CTX;
                        const float* krow = k_cache + static_cast<size_t>(kslot) * kv_dim + kv_h * cfg.head_dim;
                        attn_scores[j] = smoe::dot_f32(qhead, krow, cfg.head_dim) * attn_scale;
                    }

                    float max_val = attn_scores[0];
                    for (uint32_t j = 1; j < valid; ++j)
                        if (attn_scores[j] > max_val) max_val = attn_scores[j];
                    float sum = 0.0f;
                    for (uint32_t j = 0; j < valid; ++j) {
                        attn_scores[j] = std::exp(attn_scores[j] - max_val);
                        sum += attn_scores[j];
                    }
                    float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    float* out_head = ao + hh * cfg.head_dim;
                    for (uint32_t j = 0; j < valid; ++j) {
                        uint32_t vslot = (slot_i - j + ATTN_CTX) % ATTN_CTX;
                        const float* vrow = v_cache + static_cast<size_t>(vslot) * kv_dim + kv_h * cfg.head_dim;
                        smoe::axpy_f32(out_head, attn_scores[j] * inv_sum, vrow, cfg.head_dim);
                    }
                }

            }

            // ── ④ ALL B o_proj rows in one command buffer, then
            //     residual + FFN norm per token (CPU).
            if (P.metal) {
                for (uint32_t i = 0; i < B; ++i) {
                    g_disp_w[i]   = scout.get_o_proj(l);
                    g_disp_in[i]  = b.attn_out  + static_cast<size_t>(i) * q_dim;
                    g_disp_out[i] = b.oproj_out + static_cast<size_t>(i) * d_model;
                    g_disp_rows[i] = d_model;
                    g_disp_cols[i] = q_dim;
                }
                smoe_metal_scout_matvec_group_bf16(P.metal, g_disp_w, g_disp_in,
                                                   g_disp_out, g_disp_rows, g_disp_cols, B);
            } else {
                for (uint32_t i = 0; i < B; ++i) {
                    smoe::matvec_bf16(b.oproj_out + static_cast<size_t>(i) * d_model,
                                      scout.get_o_proj(l),
                                      b.attn_out + static_cast<size_t>(i) * q_dim,
                                      d_model, q_dim);
                }
            }
            for (uint32_t i = 0; i < B; ++i) {
                float* h  = b.hidden + static_cast<size_t>(i) * d_model;
                float* nr = b.normed + static_cast<size_t>(i) * d_model;
                const float* po = b.oproj_out + static_cast<size_t>(i) * d_model;
                for (uint32_t d = 0; d < d_model; ++d) h[d] += po[d];
                std::memcpy(nr, h, d_model * sizeof(float));
                smoe::rms_norm_bf16(nr, scout.get_post_norm(l), d_model);
            }

            // ── ⑤/⑥ FFN ───────────────────────────────────────
            if (l == 0 && cfg.has_dense_layer_0) {
                for (uint32_t i = 0; i < B; ++i) {
                    float* h  = b.hidden + static_cast<size_t>(i) * d_model;
                    float* nr = b.normed + static_cast<size_t>(i) * d_model;
                    if (P.metal) {
                        const uint16_t* weights[2] = { scout.get_l0_gate(), scout.get_l0_up() };
                        const float* inputs[2] = { nr, nr };
                        float* outputs[2] = { b.ffn_hidden, b.ffn_up };
                        uint32_t rows[2] = { cfg.ffn_dim, cfg.ffn_dim };
                        uint32_t cols[2] = { d_model, d_model };
                        smoe_metal_scout_matvec_batch_bf16(P.metal, weights, inputs, outputs, rows, cols, 2);
                    } else {
                        smoe::matvec_bf16(b.ffn_hidden, scout.get_l0_gate(), nr, cfg.ffn_dim, d_model);
                        smoe::matvec_bf16(b.ffn_up, scout.get_l0_up(), nr, cfg.ffn_dim, d_model);
                    }
                    for (uint32_t d = 0; d < cfg.ffn_dim; ++d) {
                        float val = b.ffn_hidden[d];
                        b.ffn_hidden[d] = (val / (1.0f + std::exp(-val))) * b.ffn_up[d];
                    }
                    if (P.metal) {
                        smoe_metal_scout_matvec_bf16(P.metal, scout.get_l0_down(), b.ffn_hidden, b.ffn_out, d_model, cfg.ffn_dim);
                    } else {
                        smoe::matvec_bf16(b.ffn_out, scout.get_l0_down(), b.ffn_hidden, d_model, cfg.ffn_dim);
                    }
                    for (uint32_t d = 0; d < d_model; ++d) h[d] += b.ffn_out[d];
                }
                continue;
            }

            // Routed experts: exact gate per token, then dedupe. Each
            // expert is FIRED at the streamer the moment it first enters
            // the union — the NVMe starts on this layer's blobs while
            // later tokens are still routing, instead of idling until
            // the whole chunk has routed. These are demand reads pulled
            // earlier, never extra reads (contrast the measured-slower
            // replay hints documented in main.cpp).
            const uint32_t li = l - P.moe_start_layer;
            union_reset();
            std::memset(b.routed_out, 0, static_cast<size_t>(B) * d_model * sizeof(float));

            static uint32_t topk_ids[TOPK_MAX];
            static float    topk_w[TOPK_MAX];
            for (uint32_t i = 0; i < B; ++i) {
                const float* nr = b.normed + static_cast<size_t>(i) * d_model;
                smoe::matvec_bf16(b.gate_scores, scout.get_gate(li), nr, n_experts, d_model);
                uint32_t cnt = scout.compute_top_k(b.gate_scores, n_experts, cfg.moe_top_k,
                                                   topk_ids, topk_w, cfg.norm_topk_prob);
                for (uint32_t e = 0; e < cnt; ++e) {
                    if (topk_ids[e] < MAX_EXPERTS &&
                        union_add(topk_ids[e], i, topk_w[e])) {
                        (void)streamer.prefetch(l, topk_ids[e]);
                    }
                }
            }

            // Claim each expert once; apply it to every routed token.
            uint32_t done_count = 0;
            static bool done[MAX_EXPERTS];
            std::memset(done, 0, g_union_size * sizeof(bool));
            uint64_t spin = 0;
            while (done_count < g_union_size) {
                bool progress = false;
                for (uint32_t u = 0; u < g_union_size; ++u) {
                    if (done[u]) continue;
                    const uint32_t eid = g_union_ids[u];
                    smoe::io::RingSlot* slot = streamer.claim_specific(l, eid);
                    if (!slot) {
                        if (spin > 0 && spin % 10000 == 0) (void)streamer.prefetch(l, eid);
                        continue;
                    }
                    if (P.expert_hits && l < 128 && eid < HITS_STRIDE) {
                        P.expert_hits[l * HITS_STRIDE + eid]++;
                    }
                    if (slot->data_size > 0) {
                        const uint8_t*  pg = slot->data + lay.gate_packed_offset;
                        const uint16_t* sg = reinterpret_cast<const uint16_t*>(slot->data + lay.gate_scales_offset);
                        const uint8_t*  pu = slot->data + lay.up_packed_offset;
                        const uint16_t* su = reinterpret_cast<const uint16_t*>(slot->data + lay.up_scales_offset);
                        const uint8_t*  pd = slot->data + lay.down_packed_offset;
                        const uint16_t* sd = reinterpret_cast<const uint16_t*>(slot->data + lay.down_scales_offset);

                        const ExpertTokens& et = g_elist[eid];

                        // Token-batch path: one two-pass dispatch applies
                        // this expert to every routed token at once.
                        void* bh = nullptr;
                        if (b.batch_in && b.batch_out) {
                            for (uint32_t t = 0; t < et.count; ++t) {
                                std::memcpy(b.batch_in + static_cast<size_t>(t) * d_model,
                                            b.normed + static_cast<size_t>(et.tok[t]) * d_model,
                                            d_model * sizeof(float));
                            }
                            bh = smoe_metal_fused_ffn_batch(
                                P.metal, pg, sg, pu, su, pd, sd,
                                b.batch_in, et.count,
                                lay.gate_rows, lay.gate_cols,
                                lay.down_rows, lay.down_cols,
                                P.hdr->group_size, P.hdr->bits);
                        }
                        if (bh) {
                            smoe_metal_wait_ffn_batch(P.metal, bh, b.batch_out, et.count, d_model);
                            for (uint32_t t = 0; t < et.count; ++t) {
                                float* ro = b.routed_out + static_cast<size_t>(et.tok[t]) * d_model;
                                const float* bo = b.batch_out + static_cast<size_t>(t) * d_model;
                                const float w = et.w[t];
                                for (uint32_t d = 0; d < d_model; ++d) ro[d] += w * bo[d];
                            }
                        } else {
                            // Fallback: per-token dispatch+wait
                            for (uint32_t t = 0; t < et.count; ++t) {
                                const uint32_t row = et.tok[t];
                                const float* nr = b.normed + static_cast<size_t>(row) * d_model;
                                std::memset(b.ffn_hidden, 0, cfg.ffn_dim * sizeof(float));
                                std::memset(b.ffn_out, 0, d_model * sizeof(float));
                                void* handle = smoe_metal_fused_ffn(
                                    P.metal, pg, sg, pu, su, pd, sd,
                                    nr, b.ffn_hidden, b.ffn_out,
                                    lay.gate_rows, lay.gate_cols,
                                    lay.down_rows, lay.down_cols,
                                    P.hdr->group_size, P.hdr->bits, 0);
                                if (handle) {
                                    smoe_metal_wait(P.metal, handle, b.ffn_out, 0, d_model);
                                }
                                float* ro = b.routed_out + static_cast<size_t>(row) * d_model;
                                const float w = et.w[t];
                                for (uint32_t d = 0; d < d_model; ++d) ro[d] += w * b.ffn_out[d];
                            }
                        }
                    }
                    streamer.release(slot);
                    done[u] = true;
                    done_count++;
                    progress = true;
                }
                if (!progress) {
                    std::this_thread::yield();
                    spin++;
                }
            }

            // Shared expert (DeepSeek-family; Qwen3-235B has none).
            const bool has_shared = scout.get_shared_gate(l) != nullptr && b.sh_gate != nullptr;

            for (uint32_t i = 0; i < B; ++i) {
                float* h  = b.hidden + static_cast<size_t>(i) * d_model;
                float* nr = b.normed + static_cast<size_t>(i) * d_model;
                const float* ro = b.routed_out + static_cast<size_t>(i) * d_model;

                if (has_shared) {
                    const uint32_t shared_dim = cfg.shared_expert_ffn_dim;
                    if (P.metal) {
                        const uint16_t* weights[2] = { scout.get_shared_gate(l), scout.get_shared_up(l) };
                        const float* inputs[2] = { nr, nr };
                        float* outputs[2] = { b.sh_gate, b.sh_up };
                        uint32_t rows[2] = { shared_dim, shared_dim };
                        uint32_t cols[2] = { d_model, d_model };
                        smoe_metal_scout_matvec_batch_bf16(P.metal, weights, inputs, outputs, rows, cols, 2);
                    } else {
                        smoe::matvec_bf16(b.sh_gate, scout.get_shared_gate(l), nr, shared_dim, d_model);
                        smoe::matvec_bf16(b.sh_up, scout.get_shared_up(l), nr, shared_dim, d_model);
                    }
                    for (uint32_t d = 0; d < shared_dim; ++d) {
                        float val = b.sh_gate[d];
                        b.sh_gate[d] = (val / (1.0f + std::exp(-val))) * b.sh_up[d];
                    }
                    if (P.metal) {
                        smoe_metal_scout_matvec_bf16(P.metal, scout.get_shared_down(l), b.sh_gate, b.sh_out, d_model, shared_dim);
                    } else {
                        smoe::matvec_bf16(b.sh_out, scout.get_shared_down(l), b.sh_gate, d_model, shared_dim);
                    }
                    for (uint32_t d = 0; d < d_model; ++d) h[d] += b.sh_out[d] + ro[d];
                } else {
                    for (uint32_t d = 0; d < d_model; ++d) h[d] += ro[d];
                }
            }
        }

        // Advance the ring exactly as count token-serial steps would have.
        ctx_pos = (base_slot + B) % ATTN_CTX;
        ctx_fill = (base_fill + B < ATTN_CTX) ? base_fill + B : ATTN_CTX;

        if (P.debug) {
            std::fprintf(stderr, "[prefill] chunk @%llu: %u tokens, ring pos=%u fill=%u\n",
                         static_cast<unsigned long long>(base_pos), B, ctx_pos, ctx_fill);
        }
    }
}

} // namespace smoe::prefill
