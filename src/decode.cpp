// ═══════════════════════════════════════════════════════════════
// decode.cpp — S-MoE Engine · Token-Serial Decode Step
// ═══════════════════════════════════════════════════════════════
// See decode.hpp for the design rationale. Structure per token:
//
//   embedding → RoPE tables for the absolute position →
//   for each layer L:
//     ① fused attention on GPU (one command buffer, one CPU sync) —
//        or the CPU fallback path when Metal is unavailable
//     ② exact routing: real router gate on the heavy hidden state;
//        demand prefetch of the top-k, speculative prefetch of the
//        deeper ranks through the low-priority queue
//     ③ Phase 1: every expert already resident rides ONE grouped
//        command buffer; Phase 2: shared expert (DeepSeek-family)
//        on the CPU while GPU/SSD work; Phase 3: spin-wait sweeps
//        that batch late NVMe completions into single dispatches;
//        Phase 4: wait + weighted accumulate
//   → KV ring advance
// ═══════════════════════════════════════════════════════════════

#include "decode.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace smoe::decode {

void Instr::print_and_reset() {
    if (tokens == 0) return;
    const double nt = double(tokens);
    const double accounted = dense_ms + dispatch_ms + io_spin_ms +
                             gpu_wait_ms + lm_ms;
    const double other_ms = total_ms - accounted;
    auto pct = [&](double v) {
        return total_ms > 0.0 ? 100.0 * v / total_ms : 0.0;
    };
    std::fprintf(stderr,
        "\n[instr] %llu generated tokens, %.0f ms/token (%.2f t/s)\n"
        "[instr]   dense    %8.1f ms/tok  %5.1f%%\n"
        "[instr]   dispatch %8.1f ms/tok  %5.1f%%\n"
        "[instr]   io-spin  %8.1f ms/tok  %5.1f%%\n"
        "[instr]   gpu-wait %8.1f ms/tok  %5.1f%%\n"
        "[instr]   lm-head  %8.1f ms/tok  %5.1f%%\n"
        "[instr]   other    %8.1f ms/tok  %5.1f%%\n"
        "[instr]   NVMe %.2f GB/token, effective %.2f GB/s over decode\n"
        "[instr]   adjacent-token expert overlap: %.1f%%  (%llu / %llu)\n"
        "[instr]   prev-token top-8/16/24/32 coverage of true top-8: %.1f%% / %.1f%% / %.1f%% / %.1f%%\n",
        static_cast<unsigned long long>(tokens),
        total_ms / nt, nt * 1000.0 / total_ms,
        dense_ms / nt,    pct(dense_ms),
        dispatch_ms / nt, pct(dispatch_ms),
        io_spin_ms / nt,  pct(io_spin_ms),
        gpu_wait_ms / nt, pct(gpu_wait_ms),
        lm_ms / nt,       pct(lm_ms),
        other_ms / nt,    pct(other_ms),
        double(bytes) / nt / 1e9,
        total_ms > 0.0 ? double(bytes) / 1e6 / total_ms : 0.0,
        ovl_total ? 100.0 * double(ovl_hits) / double(ovl_total) : 0.0,
        static_cast<unsigned long long>(ovl_hits),
        static_cast<unsigned long long>(ovl_total),
        cov_total ? 100.0 * double(cov[0]) / double(cov_total) : 0.0,
        cov_total ? 100.0 * double(cov[1]) / double(cov_total) : 0.0,
        cov_total ? 100.0 * double(cov[2]) / double(cov_total) : 0.0,
        cov_total ? 100.0 * double(cov[3]) / double(cov_total) : 0.0);

    dense_ms = dispatch_ms = io_spin_ms = gpu_wait_ms = lm_ms = 0.0;
    total_ms = 0.0;
    tokens = bytes = 0;
    ovl_hits = ovl_total = 0;
    cov[0] = cov[1] = cov[2] = cov[3] = 0;
    cov_total = 0;
}

void run_token_step(const Params& P, Instr* in,
                    uint32_t token, uint64_t stream_pos,
                    bool is_prompt, bool first_step,
                    uint32_t& ctx_pos, uint32_t& ctx_fill)
{
    const SmoeModelConfig& cfg = *P.cfg;
    smoe::scout::Scout& scout = *P.scout;
    smoe::io::Streamer& streamer = *P.streamer;
    const smoe::prefill::ExpertLayout& lay = *P.layout;
    const smoe::engine::Buffers& b = *P.b;
    SmoeMetalCtx* metal = P.metal;
    const bool debug = P.debug;

    const uint32_t d_model    = cfg.d_model;
    const uint32_t ffn_dim    = cfg.ffn_dim;
    const uint32_t shared_dim = cfg.shared_expert_ffn_dim;
    const uint32_t q_dim      = cfg.num_heads * cfg.head_dim;
    const uint32_t kv_dim     = cfg.num_kv_heads * cfg.head_dim;
    const uint32_t rope_half  = cfg.head_dim / 2;
    const uint32_t num_layers = cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0);

    // Buckets are charged only on instrumented generation steps.
    const bool charge = (in != nullptr) && !is_prompt;
    auto bucket = [&](double Instr::*acc) { if (charge) in->bucket(in->*acc); };

    static float attn_scores[ATTN_CTX];

    // ── Embedding ─────────────────────────────────────────────
    const uint16_t* emb = scout.get_embed() + static_cast<size_t>(token) * d_model;
    for (uint32_t d = 0; d < d_model; ++d) {
        b.hidden[d] = smoe::bf16_to_f32(emb[d]);
    }
    if (first_step && debug) {
        std::fprintf(stderr, "\n[Token %u] emb = %.4f %.4f %.4f %.4f\n",
                     token, b.hidden[0], b.hidden[1], b.hidden[2], b.hidden[3]);
    }

    // RoPE rotation for this ABSOLUTE stream position, shared by all
    // heads and layers.
    for (uint32_t d = 0; d < rope_half; ++d) {
        float angle = static_cast<float>(stream_pos) * b.rope_inv_freq[d];
        b.rope_cos[d] = std::cos(angle);
        b.rope_sin[d] = std::sin(angle);
    }

    for (uint32_t l = 0; l < num_layers; ++l) {
        std::memcpy(b.normed, b.hidden, d_model * sizeof(float));
        smoe::rms_norm_bf16(b.normed, scout.get_input_norm(l), d_model);

        float* k_cache = b.kv_cache + (l * 2 * ATTN_CTX + 0 * ATTN_CTX) * size_t(kv_dim);
        float* v_cache = b.kv_cache + (l * 2 * ATTN_CTX + 1 * ATTN_CTX) * size_t(kv_dim);
        const uint32_t slot  = ctx_pos % ATTN_CTX;
        const uint32_t valid = (ctx_fill < ATTN_CTX) ? ctx_fill + 1 : ATTN_CTX;
        const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

        if (metal) {
            // Whole attention block in ONE command buffer / ONE CPU
            // sync (was two matvec roundtrips with CPU norm/RoPE/
            // attention between them — 188 syncs/token at 94 layers).
            // o_out targets b.normed, same as the CPU path.
            SmoeAttnLayerArgs attn_args {
                scout.get_q_proj(l), scout.get_k_proj(l),
                scout.get_v_proj(l), scout.get_o_proj(l),
                scout.get_q_norm(l), scout.get_k_norm(l),
                b.rope_cos, b.rope_sin,
                b.normed,
                b.qbuf, b.kbuf, b.vbuf,
                k_cache, v_cache,
                b.attn_out, b.normed,
                d_model, q_dim, kv_dim,
                cfg.head_dim, cfg.num_heads, cfg.num_kv_heads,
                slot, valid, ATTN_CTX, scale
            };
            smoe_metal_attention_layer(metal, &attn_args);
        } else {
            smoe::matvec_bf16(b.qbuf, scout.get_q_proj(l), b.normed, q_dim, d_model);
            smoe::matvec_bf16(b.kbuf, scout.get_k_proj(l), b.normed, kv_dim, d_model);
            smoe::matvec_bf16(b.vbuf, scout.get_v_proj(l), b.normed, kv_dim, d_model);

            // Apply per-head Q/K norm
            const uint16_t* q_norm_w = scout.get_q_norm(l);
            const uint16_t* k_norm_w = scout.get_k_norm(l);
            if (q_norm_w) {
                for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                    smoe::rms_norm_bf16(b.qbuf + h * cfg.head_dim, q_norm_w, cfg.head_dim);
                }
            }
            if (k_norm_w) {
                for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                    smoe::rms_norm_bf16(b.kbuf + h * cfg.head_dim, k_norm_w, cfg.head_dim);
                }
            }

            // Apply RoPE to Q
            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                for (uint32_t d = 0; d < rope_half; ++d) {
                    float q0 = b.qbuf[h * cfg.head_dim + d];
                    float q1 = b.qbuf[h * cfg.head_dim + d + rope_half];
                    b.qbuf[h * cfg.head_dim + d]             = q0 * b.rope_cos[d] - q1 * b.rope_sin[d];
                    b.qbuf[h * cfg.head_dim + d + rope_half] = q0 * b.rope_sin[d] + q1 * b.rope_cos[d];
                }
            }

            // Apply RoPE to K
            for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                for (uint32_t d = 0; d < rope_half; ++d) {
                    float k0 = b.kbuf[h * cfg.head_dim + d];
                    float k1 = b.kbuf[h * cfg.head_dim + d + rope_half];
                    b.kbuf[h * cfg.head_dim + d]             = k0 * b.rope_cos[d] - k1 * b.rope_sin[d];
                    b.kbuf[h * cfg.head_dim + d + rope_half] = k0 * b.rope_sin[d] + k1 * b.rope_cos[d];
                }
            }

            std::memcpy(k_cache + slot * size_t(kv_dim), b.kbuf, kv_dim * sizeof(float));
            std::memcpy(v_cache + slot * size_t(kv_dim), b.vbuf, kv_dim * sizeof(float));

            // Multi-Head Attention CPU (with GQA support)
            std::memset(b.attn_out, 0, q_dim * sizeof(float));

            uint32_t heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                uint32_t kv_h = h / heads_per_kv;
                const float* qhead = b.qbuf + h * cfg.head_dim;
                for (uint32_t i = 0; i < valid; ++i) {
                    uint32_t ki = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                    const float* krow = k_cache + ki * size_t(kv_dim) + kv_h * cfg.head_dim;
                    attn_scores[i] = smoe::dot_f32(qhead, krow, cfg.head_dim) * scale;
                }

                float max_val = attn_scores[0];
                for (uint32_t i = 1; i < valid; ++i) {
                    if (attn_scores[i] > max_val) max_val = attn_scores[i];
                }
                float sum = 0.0f;
                for (uint32_t i = 0; i < valid; ++i) {
                    attn_scores[i] = std::exp(attn_scores[i] - max_val);
                    sum += attn_scores[i];
                }
                float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (uint32_t i = 0; i < valid; ++i) {
                    attn_scores[i] *= inv_sum;
                }

                float* out_head = b.attn_out + h * cfg.head_dim;
                for (uint32_t i = 0; i < valid; ++i) {
                    uint32_t vi = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                    smoe::axpy_f32(out_head, attn_scores[i],
                                   v_cache + vi * size_t(kv_dim) + kv_h * cfg.head_dim,
                                   cfg.head_dim);
                }
            }

            // O Proj
            smoe::matvec_bf16(b.normed, scout.get_o_proj(l), b.attn_out, d_model, q_dim);
        }
        for (uint32_t i = 0; i < d_model; ++i) {
            b.hidden[i] += b.normed[i];
        }

        // FFN Norm
        std::memcpy(b.normed, b.hidden, d_model * sizeof(float));
        smoe::rms_norm_bf16(b.normed, scout.get_post_norm(l), d_model);

        // FFN
        if (l == 0 && cfg.has_dense_layer_0) {
            if (metal) {
                const uint16_t* weights[2] = { scout.get_l0_gate(), scout.get_l0_up() };
                const float* inputs[2] = { b.normed, b.normed };
                float* outputs[2] = { b.l0_gate, b.l0_up };
                uint32_t rows[2] = { ffn_dim, ffn_dim };
                uint32_t cols[2] = { d_model, d_model };
                smoe_metal_scout_matvec_batch_bf16(metal, weights, inputs, outputs, rows, cols, 2);
            } else {
                smoe::matvec_bf16(b.l0_gate, scout.get_l0_gate(), b.normed, ffn_dim, d_model);
                smoe::matvec_bf16(b.l0_up, scout.get_l0_up(), b.normed, ffn_dim, d_model);
            }
            for (uint32_t i = 0; i < ffn_dim; ++i) {
                // Silu
                float val = b.l0_gate[i];
                b.l0_gate[i] = (val / (1.0f + std::exp(-val))) * b.l0_up[i];
            }
            if (metal) {
                smoe_metal_scout_matvec_bf16(metal, scout.get_l0_down(), b.l0_gate, b.normed, d_model, ffn_dim);
            } else {
                smoe::matvec_bf16(b.normed, scout.get_l0_down(), b.l0_gate, d_model, ffn_dim);
            }
            for (uint32_t d = 0; d < d_model; ++d) b.hidden[d] += b.normed[d];
            if (first_step && debug) {
                std::fprintf(stderr, "\n[L0] hidden = %.4f %.4f %.4f %.4f\n", b.hidden[0], b.hidden[1], b.hidden[2], b.hidden[3]);
            }
        } else {
            // Exact routing, ALL steps: the real router gate evaluated
            // on the heavy hidden state. One ranked top-(k+spec)
            // selection, k = cfg.moe_top_k from the vault arch block
            // (8 for Qwen3): ranks 0..k-1 are the true routing for THIS
            // token (weights renormalised over the k per norm_topk_prob —
            // identical to a direct top-k selection); ranks beyond k
            // are the speculative prefetch bet for the NEXT token at
            // this layer (measured coverage of the next token's top-8:
            // 46.4% retention alone → 63.7% at +8 → 71.9% at +16).
            // Speculative reads use the streamer's low-priority queue
            // so they can never delay the demand fetches this token is
            // spinning on.
            smoe::matvec_bf16(b.gate_scores,
                              scout.get_gate(l - P.moe_start_layer),
                              b.normed,
                              cfg.max_experts_per_layer, d_model);
            // Rank buffers are 32 wide (top_k 16 max + spec 24 max ⇒ clamp).
            static smoe::scout::ExpertPrediction pred_state {};
            static uint32_t rank_ids[32];
            static float    rank_w[32];
            uint32_t want = cfg.moe_top_k + P.spec_width;
            if (want > 32) want = 32;
            const uint32_t nsel = scout.compute_top_k(
                b.gate_scores, cfg.max_experts_per_layer, want,
                rank_ids, rank_w, false);
            const uint32_t nk = (nsel < cfg.moe_top_k) ? nsel : cfg.moe_top_k;
            pred_state.layer_id = l;
            pred_state.count = nk;
            float w_sum = 0.0f;
            for (uint32_t e = 0; e < nk; ++e) w_sum += rank_w[e];
            const float w_inv =
                (cfg.norm_topk_prob && w_sum > 0.0f) ? 1.0f / w_sum : 1.0f;
            for (uint32_t e = 0; e < nk; ++e) {
                pred_state.expert_ids[e]     = rank_ids[e];
                pred_state.expert_weights[e] = rank_w[e] * w_inv;
                (void)streamer.prefetch(l, rank_ids[e]);
            }
            for (uint32_t e = nk; e < nsel; ++e) {
                (void)streamer.prefetch(l, rank_ids[e], /*speculative=*/true);
            }
            const smoe::scout::ExpertPrediction& pred = pred_state;

            // Rank coverage: would prefetching the PREVIOUS token's
            // top-M gate ranking have covered this token's true top-8?
            // Decides the width of a scout-free speculative prefetch.
            if (in && l < 128) {
                if (!is_prompt) {
                    for (uint32_t e = 0; e < pred.count; ++e) {
                        for (uint32_t r = 0; r < in->prev_top32_n[l]; ++r) {
                            if (in->prev_top32[l][r] == pred.expert_ids[e]) {
                                if (r < 8)  in->cov[0]++;
                                if (r < 16) in->cov[1]++;
                                if (r < 24) in->cov[2]++;
                                in->cov[3]++;   // r < 32 by construction
                                break;
                            }
                        }
                    }
                    in->cov_total += pred.count;
                }
                static uint32_t t32_ids[32];
                static float    t32_w[32];
                uint32_t n32 = scout.compute_top_k(
                    b.gate_scores, cfg.max_experts_per_layer, 32,
                    t32_ids, t32_w, false);
                if (n32 > 32) n32 = 32;
                in->prev_top32_n[l] = static_cast<uint8_t>(n32);
                std::memcpy(in->prev_top32[l], t32_ids, n32 * sizeof(uint32_t));
            }

            // ── Phase 1: Dispatch Currently Available Routed Experts to GPU ──
            bucket(&Instr::dense_ms);   // norms+QKV+attention+o_proj+gate
            std::memset(b.routed_out, 0, d_model * sizeof(float));

            // Adjacent-token routed-expert overlap: how many of this
            // token's experts were also routed by the previous token —
            // the upper bound on what cross-token ring retention buys.
            // Prompt steps seed the masks without counting.
            if (in && l < 128) {
                uint64_t cur0 = 0, cur1 = 0;
                for (uint32_t e = 0; e < pred.count; ++e) {
                    const uint32_t id = pred.expert_ids[e];
                    if (id < 64)        cur0 |= 1ULL << id;
                    else if (id < 128)  cur1 |= 1ULL << (id - 64);
                }
                if (!is_prompt) {
                    in->ovl_hits  += __builtin_popcountll(cur0 & in->prev_mask[l][0])
                                   + __builtin_popcountll(cur1 & in->prev_mask[l][1]);
                    in->ovl_total += pred.count;
                }
                in->prev_mask[l][0] = cur0;
                in->prev_mask[l][1] = cur1;
            }
            bool executed[16] = {false};
            uint32_t num_executed = 0;
            uint64_t spin = 0;

            void* wait_handles[16] = {nullptr};
            smoe::io::RingSlot* active_slots[16] = {nullptr};

            // Grouped dispatch staging: every expert whose slot is
            // ready in the same sweep rides ONE command buffer
            // (was: one command buffer per expert — 752/token).
            SmoeExpertBlob group_blobs[16];
            uint32_t group_n = 0;
            auto stage_expert = [&](smoe::io::RingSlot* slot, uint32_t e) {
                group_blobs[group_n++] = SmoeExpertBlob{
                    slot->data + lay.gate_packed_offset,
                    reinterpret_cast<const uint16_t*>(slot->data + lay.gate_scales_offset),
                    slot->data + lay.up_packed_offset,
                    reinterpret_cast<const uint16_t*>(slot->data + lay.up_scales_offset),
                    slot->data + lay.down_packed_offset,
                    reinterpret_cast<const uint16_t*>(slot->data + lay.down_scales_offset),
                    e
                };
            };
            auto dispatch_group = [&]() {
                if (group_n == 0) return;
                void* h = smoe_metal_fused_ffn_group(
                    metal, group_blobs, group_n, b.normed,
                    lay.gate_rows, lay.gate_cols,
                    lay.down_rows, lay.down_cols,
                    P.hdr->group_size, P.hdr->bits);
                for (uint32_t i = 0; i < group_n; ++i)
                    wait_handles[group_blobs[i].slot] = h;
                group_n = 0;
            };

            // Initial non-blocking pass
            if (debug) {
                std::fprintf(stderr, "[DEBUG] layer %u: pred.count = %u. Experts: ", l, pred.count);
                for (uint32_t e = 0; e < pred.count; ++e) {
                    std::fprintf(stderr, "%u ", pred.expert_ids[e]);
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }

            for (uint32_t e = 0; e < pred.count; ++e) {
                smoe::io::RingSlot* slot = streamer.claim_specific(pred.layer_id, pred.expert_ids[e]);
                if (slot) {
                    active_slots[e] = slot;
                    if (pred.layer_id < 128 && pred.expert_ids[e] < smoe::prefill::HITS_STRIDE)
                        P.expert_hits[pred.layer_id * smoe::prefill::HITS_STRIDE + pred.expert_ids[e]]++;
                    if (slot->data_size > 0) stage_expert(slot, e);
                    executed[e] = true;
                    num_executed++;
                }
            }
            dispatch_group();   // all ring hits in one command buffer
            bucket(&Instr::dispatch_ms);

            // ── Phase 2: Compute Shared Expert on CPU while GPU/SSD works ──
            std::memset(b.shared_out, 0, d_model * sizeof(float));

            if (scout.get_shared_gate(l)) {
                if (metal) {
                    const uint16_t* weights[2] = { scout.get_shared_gate(l), scout.get_shared_up(l) };
                    const float* inputs[2] = { b.normed, b.normed };
                    float* outputs[2] = { b.sh_gate, b.sh_up };
                    uint32_t rows[2] = { shared_dim, shared_dim };
                    uint32_t cols[2] = { d_model, d_model };
                    smoe_metal_scout_matvec_batch_bf16(metal, weights, inputs, outputs, rows, cols, 2);
                } else {
                    smoe::matvec_bf16(b.sh_gate, scout.get_shared_gate(l), b.normed, shared_dim, d_model);
                    smoe::matvec_bf16(b.sh_up, scout.get_shared_up(l), b.normed, shared_dim, d_model);
                }
                for (uint32_t i = 0; i < shared_dim; ++i) {
                    float val = b.sh_gate[i];
                    b.sh_gate[i] = (val / (1.0f + std::exp(-val))) * b.sh_up[i];
                }
                if (metal) {
                    smoe_metal_scout_matvec_bf16(metal, scout.get_shared_down(l), b.sh_gate, b.shared_out, d_model, shared_dim);
                } else {
                    smoe::matvec_bf16(b.shared_out, scout.get_shared_down(l), b.sh_gate, d_model, shared_dim);
                }
            }

            bucket(&Instr::dense_ms);   // shared expert (none on Qwen3)

            // ── Phase 3: Spin-wait for any remaining missing experts ──
            // Each sweep groups everything that became READY since
            // the last one into a single command buffer, so a burst
            // of completed reads costs one commit, not one each.
            while (num_executed < pred.count) {
                bool made_progress = false;
                for (uint32_t e = 0; e < pred.count; ++e) {
                    if (executed[e]) continue;

                    smoe::io::RingSlot* slot = streamer.claim_specific(pred.layer_id, pred.expert_ids[e]);
                    if (slot) {
                        active_slots[e] = slot;
                        if (pred.layer_id < 128 && pred.expert_ids[e] < smoe::prefill::HITS_STRIDE)
                            P.expert_hits[pred.layer_id * smoe::prefill::HITS_STRIDE + pred.expert_ids[e]]++;
                        if (slot->data_size > 0) stage_expert(slot, e);
                        executed[e] = true;
                        num_executed++;
                        made_progress = true;
                    } else {
                        if (spin % 10000 == 0) {
                            (void)streamer.prefetch(pred.layer_id, pred.expert_ids[e]);
                        }
                    }
                }
                dispatch_group();

                if (!made_progress) {
                    std::this_thread::yield();
                    spin++;
                }
            }
            bucket(&Instr::io_spin_ms);

            // ── Phase 4: Wait for GPU and accumulate ──
            for (uint32_t e = 0; e < pred.count; ++e) {
                if (wait_handles[e]) {
                    float* row = b.expert_out + size_t(e) * d_model;
                    smoe_metal_wait(metal, wait_handles[e], row, e, d_model);
                    const float weight = pred.expert_weights[e];
                    if (debug) {
                        if (l == 1 && first_step) {
                            std::fprintf(stderr, "[L1] Routed Expert %u, weight = %.4f\n", pred.expert_ids[e], weight);
                        }
                        std::fprintf(stderr, "[MoE_OUT] layer=%u e=%u sum_first=%f\n", l, e, row[0]);
                    }
                    smoe::axpy_f32(b.routed_out, weight, row, d_model);
                }
                if (active_slots[e]) {
                    streamer.release(active_slots[e]);
                }
            }
            bucket(&Instr::gpu_wait_ms);

            if (l == 1 && first_step && debug) {
                std::fprintf(stderr, "\n[L1] shared_out = %.4f %.4f %.4f %.4f\n", b.shared_out[0], b.shared_out[1], b.shared_out[2], b.shared_out[3]);
                std::fprintf(stderr, "[L1] routed_out = %.4f %.4f %.4f %.4f\n", b.routed_out[0], b.routed_out[1], b.routed_out[2], b.routed_out[3]);
            }

            // Add FFN residuals
            for (uint32_t d = 0; d < d_model; ++d) {
                b.hidden[d] += b.shared_out[d] + b.routed_out[d];
            }

            if (l == 1 && first_step && debug) {
                std::fprintf(stderr, "\n[L1] hidden after FFN = %.4f %.4f %.4f %.4f\n", b.hidden[0], b.hidden[1], b.hidden[2], b.hidden[3]);
            }
        }
    }

    ctx_pos = (ctx_pos + 1) % ATTN_CTX;
    if (ctx_fill < ATTN_CTX) ++ctx_fill;
}

} // namespace smoe::decode
