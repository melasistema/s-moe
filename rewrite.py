import re

code = """ScoutOutput Scout::Impl::neural_forward(uint32_t token_id) noexcept {
    ScoutOutput out {};

    // ── 1. Token embedding lookup ─────────────────────────────
    const uint32_t safe_id = (token_id < VOCAB_SIZE) ? token_id : 0;
    const float*   emb_row = w_embed + static_cast<size_t>(safe_id) * D_MODEL;
    std::memcpy(hidden, emb_row, D_MODEL * sizeof(float));

    // ── 2. Full 28-Layer Backbone Execution ───────────────────
    for (uint32_t l = 0; l <= NUM_MOE_LAYERS; ++l) {
        
        // a. Input RMS LayerNorm
        std::memcpy(normed, hidden, D_MODEL * sizeof(float));
        rms_norm(normed, w_input_norm[l], D_MODEL);

        // b. Q/K/V projections
        if (metal_ctx) {
            const float* weights[3] = { w_q_proj[l], w_k_proj[l], w_v_proj[l] };
            const float* inputs[3]  = { normed, normed, normed };
            float* outputs[3]       = { qbuf, kbuf, vbuf };
            uint32_t rows[3]        = { D_MODEL, D_MODEL, D_MODEL };
            uint32_t cols[3]        = { D_MODEL, D_MODEL, D_MODEL };
            smoe_metal_scout_matvec_batch(metal_ctx, weights, inputs, outputs, rows, cols, 3);
        } else {
            matvec(qbuf, w_q_proj[l], normed, D_MODEL, D_MODEL);
            matvec(kbuf, w_k_proj[l], normed, D_MODEL, D_MODEL);
            matvec(vbuf, w_v_proj[l], normed, D_MODEL, D_MODEL);
        }

        // Apply RoPE
        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t d = 0; d < 64; ++d) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(d * 2) / 128.0f);
                float angle = static_cast<float>(step) * freq;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                float q0 = qbuf[h * 128 + d];
                float q1 = qbuf[h * 128 + d + 64];
                qbuf[h * 128 + d]      = q0 * cos_val - q1 * sin_val;
                qbuf[h * 128 + d + 64] = q0 * sin_val + q1 * cos_val;
                
                float k0 = kbuf[h * 128 + d];
                float k1 = kbuf[h * 128 + d + 64];
                kbuf[h * 128 + d]      = k0 * cos_val - k1 * sin_val;
                kbuf[h * 128 + d + 64] = k0 * sin_val + k1 * cos_val;
            }
        }

        // c. Write K/V into ring cache for this layer
        const uint32_t slot = ctx_pos % ATTN_CTX;
        float* layer_k = k_cache + (l * ATTN_CTX + slot) * D_MODEL;
        float* layer_v = v_cache + (l * ATTN_CTX + slot) * D_MODEL;
        std::memcpy(layer_k, kbuf, D_MODEL * sizeof(float));
        std::memcpy(layer_v, vbuf, D_MODEL * sizeof(float));

        // d. Scaled dot-product MHA
        const float scale = 1.0f / std::sqrt(128.0f);
        const uint32_t valid = (ctx_fill < ATTN_CTX) ? ctx_fill + 1 : ATTN_CTX;
        std::memset(attn_out, 0, D_MODEL * sizeof(float));

        for (uint32_t h = 0; h < 16; ++h) {
            for (uint32_t i = 0; i < valid; ++i) {
                uint32_t ki = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                float dot_qk = 0.0f;
                const float* krow = k_cache + (l * ATTN_CTX + ki) * D_MODEL + h * 128;
                const float* qhead = qbuf + h * 128;
                for (uint32_t d = 0; d < 128; ++d) {
                    dot_qk += qhead[d] * krow[d];
                }
                attn_scores[i] = dot_qk * scale;
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

            for (uint32_t i = 0; i < valid; ++i) {
                uint32_t vi = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                const float alpha = attn_scores[i];
                const float* vrow = v_cache + (l * ATTN_CTX + vi) * D_MODEL + h * 128;
                float* out_head = attn_out + h * 128;
                for (uint32_t d = 0; d < 128; ++d) {
                    out_head[d] += alpha * vrow[d];
                }
            }
        }

        // e. Output projection + residual
        if (metal_ctx) {
            smoe_metal_scout_matvec(metal_ctx, w_o_proj[l], attn_out, normed, D_MODEL, D_MODEL);
        } else {
            matvec(normed, w_o_proj[l], attn_out, D_MODEL, D_MODEL);
        }
        for (uint32_t d = 0; d < D_MODEL; ++d) hidden[d] += normed[d];

        // f. Post-attention RMS norm
        std::memcpy(normed, hidden, D_MODEL * sizeof(float));
        rms_norm(normed, w_post_norm[l], D_MODEL);

        // g. Heavy Expert Gate Prediction (for l >= 1)
        if (l >= 1) {
            ExpertPrediction& pred = out.routing[l - 1];
            pred.layer_id = l;
            float* scores = gate_scores_batch + (l - 1) * GATE_ROWS;
            
            if (metal_ctx) {
                smoe_metal_scout_matvec(metal_ctx, gate_w[l - 1], normed, scores, GATE_ROWS, D_MODEL);
            } else {
                matvec(scores, gate_w[l - 1], normed, GATE_ROWS, D_MODEL);
            }
            softmax(scores, GATE_ROWS);
            pred.count = top_k(scores, GATE_ROWS, MAX_ACTIVE, pred.expert_ids, pred.expert_weights);
        }

        // h. Dense / Shared Experts FFN
        if (l == 0) {
            static float l0_gate_out[10944];
            static float l0_up_out[10944];
            if (metal_ctx) {
                const float* weights[2] = { w_l0_gate, w_l0_up };
                const float* inputs[2]  = { normed, normed };
                float* outputs[2]       = { l0_gate_out, l0_up_out };
                uint32_t rows[2]        = { 10944, 10944 };
                uint32_t cols[2]        = { D_MODEL, D_MODEL };
                smoe_metal_scout_matvec_batch(metal_ctx, weights, inputs, outputs, rows, cols, 2);
            } else {
                matvec(l0_gate_out, w_l0_gate, normed, 10944, D_MODEL);
                matvec(l0_up_out, w_l0_up, normed, 10944, D_MODEL);
            }
            for (uint32_t i = 0; i < 10944; ++i) {
                float val = l0_gate_out[i];
                l0_gate_out[i] = (val / (1.0f + std::exp(-val))) * l0_up_out[i];
            }
            if (metal_ctx) smoe_metal_scout_matvec(metal_ctx, w_l0_down, l0_gate_out, normed, D_MODEL, 10944);
            else matvec(normed, w_l0_down, l0_gate_out, D_MODEL, 10944);
            for (uint32_t d = 0; d < D_MODEL; ++d) hidden[d] += normed[d];
        } else {
            static float shared_gate_out[2816];
            static float shared_up_out[2816];
            if (metal_ctx) {
                const float* weights[2] = { w_shared_gate[l], w_shared_up[l] };
                const float* inputs[2]  = { normed, normed };
                float* outputs[2]       = { shared_gate_out, shared_up_out };
                uint32_t rows[2]        = { 2816, 2816 };
                uint32_t cols[2]        = { D_MODEL, D_MODEL };
                smoe_metal_scout_matvec_batch(metal_ctx, weights, inputs, outputs, rows, cols, 2);
            } else {
                matvec(shared_gate_out, w_shared_gate[l], normed, 2816, D_MODEL);
                matvec(shared_up_out, w_shared_up[l], normed, 2816, D_MODEL);
            }
            for (uint32_t i = 0; i < 2816; ++i) {
                float val = shared_gate_out[i];
                shared_gate_out[i] = (val / (1.0f + std::exp(-val))) * shared_up_out[i];
            }
            if (metal_ctx) smoe_metal_scout_matvec(metal_ctx, w_shared_down[l], shared_gate_out, normed, D_MODEL, 2816);
            else matvec(normed, w_shared_down[l], shared_gate_out, D_MODEL, 2816);
            for (uint32_t d = 0; d < D_MODEL; ++d) hidden[d] += normed[d];
        }
    }

    ctx_pos = (ctx_pos + 1) % ATTN_CTX;
    if (ctx_fill < ATTN_CTX) ++ctx_fill;

    // ── 3. Final model norm ───────────────────────────────────
    rms_norm(hidden, w_model_norm, D_MODEL);

    // ── 4. LM Head ────────────────────────────────────────────
    if (metal_ctx) {
        smoe_metal_scout_matvec(metal_ctx, w_lm_head, hidden, lm_head_scores, VOCAB_SIZE, D_MODEL);
    } else {
        for (uint32_t v = 0; v < VOCAB_SIZE; ++v) {
            const float* row = w_lm_head + static_cast<size_t>(v) * D_MODEL;
            float score = 0.0f;
            for (uint32_t d = 0; d < D_MODEL; ++d) score += row[d] * hidden[d];
            lm_head_scores[v] = score;
        }
    }

    uint32_t best_tok   = 0;
    float    best_score = -1e38f;
    for (uint32_t v = 0; v < VOCAB_SIZE; ++v) {
        if (lm_head_scores[v] > best_score) {
            best_score = lm_head_scores[v];
            best_tok   = v;
        }
    }
    out.next_token_id = best_tok;

    ++step;
    return out;
}"""

with open('src/scout/scout.cpp', 'r') as f:
    text = f.read()

start_idx = text.find("ScoutOutput Scout::Impl::neural_forward")
end_idx = text.find("uint32_t Scout::Impl::ngram_hash")

if start_idx != -1 and end_idx != -1:
    end_idx = text.rfind("// ════════════════════════════════", start_idx, end_idx)
    new_text = text[:start_idx] + code + "\n\n" + text[end_idx:]
    with open('src/scout/scout.cpp', 'w') as f:
        f.write(new_text)
    print("Successfully replaced neural_forward.")
else:
    print("Could not find start or end indices.")
