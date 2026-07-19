// ═══════════════════════════════════════════════════════════════
// scout.cpp — S-MoE Engine · Surface Scout (resident backbone)
// ═══════════════════════════════════════════════════════════════
// The Surface Scout is the heavy model's own dense trunk — the
// ~16 GB of bf16 tensors the sculptor did not exile into the
// expert vault: embeddings, attention projections, norms, router
// gates, LM head. This file does three jobs:
//
//   ① Parse the safetensors header (hand-rolled, no JSON library)
//      and wire weight pointers directly into the mmap region.
//   ② Auto-detect model topology (GQA shape, QK-norm, dense L0,
//      shared experts) from the tensor names present.
//   ③ Serve those weights to the engine through const accessors,
//      plus top-k expert selection for exact routing.
//
// Memory invariants:
//   ① The safetensors file is mmap()-ed MAP_SHARED and stays
//     resident (a background page-toucher prefaults it once).
//     F_NOCACHE is NOT applied here — that is reserved for the
//     vault streamer, which must bypass the page cache.
//   ② The only runtime allocation is the lm_head scores scratch,
//     made once in the constructor.
//
// History: this class once ran a full speculative forward pass
// (neural + Week-4 heuristic oracle) to predict expert routing
// ahead of the heavy model. Measurement retired it — 51.5% oracle
// accuracy at ~265 ms/token vs 46.4% free coverage from ring
// retention — and with it went the forward scratch buffers, the
// ~1.6 GB KV mirror, and the heuristic tables. The Scout is now
// an artifact, not an agent.
// ═══════════════════════════════════════════════════════════════

#include "scout.hpp"
#include "../compute/metal_bridge.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <thread>

// POSIX / macOS headers for mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace smoe::scout {

static void info_printf(const char* format, ...) {
    static const bool enabled = [] {
        const char* sm_dbg = std::getenv("SMOE_DEBUG");
        return sm_dbg && (std::strcmp(sm_dbg, "1") == 0 ||
                          std::strcmp(sm_dbg, "true") == 0);
    }();
    if (!enabled) return;
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

// ═══════════════════════════════════════════════════════════════
// §1  Top-k expert selection
// ═══════════════════════════════════════════════════════════════

// Select top-k indices from scores[n] into out_indices[k].
// Returns k (or fewer if n < k).
// Uses a simple partial selection — fine for k=8, n=128.
//
// norm_topk: if true, re-normalize the selected k weights to sum=1.0
//   (Qwen3 norm_topk_prob=true; DeepSeek uses softmax-over-all pre-normalised weights)
static uint32_t top_k(const float* scores, uint32_t n, uint32_t k,
                       uint32_t* out_indices, float* out_weights = nullptr,
                       bool norm_topk = false) noexcept {
    uint32_t count = std::min(k, n);
    float* tmp = (float*)__builtin_alloca(n * sizeof(float));
    uint32_t* idx = (uint32_t*)__builtin_alloca(n * sizeof(uint32_t));

    // Softmax over all experts to get proper routing probabilities
    float max_val = scores[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (scores[i] > max_val) max_val = scores[i];
    }
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        tmp[i] = std::exp(scores[i] - max_val);
        idx[i] = i;
        sum_exp += tmp[i];
    }
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        tmp[i] *= inv_sum;
    }

    // Top-k selection (partial selection sort)
    for (uint32_t s = 0; s < count; ++s) {
        uint32_t best = s;
        for (uint32_t i = s + 1; i < n; ++i) {
            if (tmp[i] > tmp[best]) best = i;
        }
        float   ft = tmp[s]; tmp[s] = tmp[best]; tmp[best] = ft;
        uint32_t it = idx[s]; idx[s] = idx[best]; idx[best] = it;
        out_indices[s] = idx[s];
        if (out_weights) out_weights[s] = tmp[s];
    }

    // Re-normalize selected weights to sum=1.0 when norm_topk_prob=true (Qwen3)
    if (norm_topk && out_weights) {
        float topk_sum = 0.0f;
        for (uint32_t s = 0; s < count; ++s) topk_sum += out_weights[s];
        if (topk_sum > 0.0f) {
            float inv = 1.0f / topk_sum;
            for (uint32_t s = 0; s < count; ++s) out_weights[s] *= inv;
        }
    }

    return count;
}

// ═══════════════════════════════════════════════════════════════
// §2  Minimal safetensors header parser
// ═══════════════════════════════════════════════════════════════
// Parses just enough of the JSON header to extract:
//   {"tensor_name": {"dtype": "BF16", "shape": [...], "data_offsets": [start, end]}}
//
// No external JSON library — hand-rolled scanner that locates
// the fields we need by string searching.  This is intentionally
// minimal and reads only the tensors we care about.

struct TensorMeta {
    uint64_t data_start { 0 };   // offset from data region start
    uint64_t data_end   { 0 };
    uint32_t shape[4]   { 0, 0, 0, 0 };
    uint32_t ndim       { 0 };
    bool     valid      { false };
};

// Advance past whitespace
static const char* skip_ws(const char* p, const char* end) noexcept {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    return p;
}

// Parse a non-negative 64-bit integer at p; advance p past it.
static uint64_t parse_uint64(const char*& p, const char* end) noexcept {
    uint64_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        v = v * 10 + static_cast<uint64_t>(*p - '0');
        ++p;
    }
    return v;
}

// Find the first occurrence of needle in [haystack, end).
// Returns pointer to it, or nullptr.
static const char* find_str(const char* haystack, const char* end,
                             const char* needle) noexcept {
    size_t nlen = std::strlen(needle);
    while (haystack + nlen <= end) {
        if (std::memcmp(haystack, needle, nlen) == 0) return haystack;
        ++haystack;
    }
    return nullptr;
}

// Parse one tensor's metadata block from the JSON header.
// We locate the tensor_name key, then extract data_offsets and shape.
static TensorMeta parse_tensor_meta(const char* json_start,
                                    const char* json_end,
                                    const char* tensor_name) noexcept {
    TensorMeta m;

    // Build search key: "tensor_name":
    // We scan for the exact quoted name.
    // Max tensor name length: 128 chars
    char key[160];
    int  key_len = std::snprintf(key, sizeof(key), "\"%s\"", tensor_name);
    if (key_len <= 0 || key_len >= static_cast<int>(sizeof(key))) return m;

    const char* p = find_str(json_start, json_end, key);
    if (!p) return m;
    p += key_len;

    // Now expect :  {  ...  "data_offsets": [start, end]  ...  }
    // Advance to the opening brace of this tensor's object
    p = skip_ws(p, json_end);
    if (p >= json_end || *p != ':') return m;
    ++p;
    p = skip_ws(p, json_end);
    if (p >= json_end || *p != '{') return m;

    // Find the closing brace of this object (no nested objects in safetensors)
    const char* obj_end = p + 1;
    while (obj_end < json_end && *obj_end != '}') ++obj_end;
    if (obj_end >= json_end) return m;
    ++obj_end; // exclusive end

    // ── data_offsets ─────────────────────────────────────────
    const char* do_ptr = find_str(p, obj_end, "\"data_offsets\"");
    if (!do_ptr) return m;
    do_ptr += std::strlen("\"data_offsets\"");
    do_ptr = skip_ws(do_ptr, obj_end);
    if (do_ptr >= obj_end || *do_ptr != ':') return m;
    ++do_ptr;
    do_ptr = skip_ws(do_ptr, obj_end);
    if (do_ptr >= obj_end || *do_ptr != '[') return m;
    ++do_ptr;
    do_ptr = skip_ws(do_ptr, obj_end);
    m.data_start = parse_uint64(do_ptr, obj_end);
    do_ptr = skip_ws(do_ptr, obj_end);
    if (do_ptr >= obj_end || *do_ptr != ',') return m;
    ++do_ptr;
    do_ptr = skip_ws(do_ptr, obj_end);
    m.data_end = parse_uint64(do_ptr, obj_end);

    // ── shape ─────────────────────────────────────────────────
    const char* sh_ptr = find_str(p, obj_end, "\"shape\"");
    if (!sh_ptr) return m;
    sh_ptr += std::strlen("\"shape\"");
    sh_ptr = skip_ws(sh_ptr, obj_end);
    if (sh_ptr >= obj_end || *sh_ptr != ':') return m;
    ++sh_ptr;
    sh_ptr = skip_ws(sh_ptr, obj_end);
    if (sh_ptr >= obj_end || *sh_ptr != '[') return m;
    ++sh_ptr;
    m.ndim = 0;
    while (sh_ptr < obj_end && *sh_ptr != ']' && m.ndim < 4) {
        sh_ptr = skip_ws(sh_ptr, obj_end);
        if (sh_ptr >= obj_end || *sh_ptr == ']') break;
        m.shape[m.ndim++] = static_cast<uint32_t>(parse_uint64(sh_ptr, obj_end));
        sh_ptr = skip_ws(sh_ptr, obj_end);
        if (sh_ptr < obj_end && *sh_ptr == ',') ++sh_ptr;
    }

    m.valid = true;
    return m;
}

// ═══════════════════════════════════════════════════════════════
// §3  Scout::Impl
// ═══════════════════════════════════════════════════════════════

struct Scout::Impl {
    // ── Model configuration ───────────────────────────────────
    SmoeModelConfig cfg {};

    // ── Load-success flag ──────────────────────────────────────
    bool neural_mode { false };   // true ↔ safetensors loaded

    // ── Metal context ─────────────────────────────────────────
    SmoeMetalCtx* metal_ctx { nullptr };

    // ── mmap state ────────────────────────────────────────────
    void*    mmap_base { nullptr };
    size_t   mmap_size { 0 };

    // Background page-toucher that streams the mapping in sequentially;
    // joined in the destructor before munmap.
    std::thread prefault_thread;

    // ── Weight tensor pointers (bf16, into the mmap region) ───

    // Embedding table / LM head: [cfg.vocab_size × cfg.d_model]
    const uint16_t* w_embed      { nullptr };
    const uint16_t* w_lm_head    { nullptr };

    // Layers 0..N attention weights
    const uint16_t* w_q_proj[MAX_MOE_LAYERS] {};
    const uint16_t* w_k_proj[MAX_MOE_LAYERS] {};
    const uint16_t* w_v_proj[MAX_MOE_LAYERS] {};
    const uint16_t* w_o_proj[MAX_MOE_LAYERS] {};

    // Layers 0..N norms
    const uint16_t* w_input_norm[MAX_MOE_LAYERS] {};
    const uint16_t* w_post_norm[MAX_MOE_LAYERS]  {};

    // Layer 0 dense MLP
    const uint16_t* w_l0_gate { nullptr };
    const uint16_t* w_l0_up   { nullptr };
    const uint16_t* w_l0_down { nullptr };

    // Layers 1..N shared experts (nullptr = model has no shared expert, e.g. Qwen3-235B)
    const uint16_t* w_shared_gate[MAX_MOE_LAYERS] {};
    const uint16_t* w_shared_up[MAX_MOE_LAYERS]   {};
    const uint16_t* w_shared_down[MAX_MOE_LAYERS] {};

    // Qwen3-specific: per-head Q/K RMS norms applied before RoPE
    // Shape [head_dim] each, nullptr if model doesn't use QK-norm (DeepSeek)
    const uint16_t* w_q_norm[MAX_MOE_LAYERS] {};
    const uint16_t* w_k_norm[MAX_MOE_LAYERS] {};

    // Final norm: [cfg.d_model]
    const uint16_t* w_model_norm { nullptr };

    // Router gate weights for MoE layers: gate_w[l] for MoE layer l
    // (offset by has_dense_layer_0 in the loader)
    const uint16_t* gate_w[MAX_MOE_LAYERS] {};

    // LM head score buffer [cfg.vocab_size] — the engine computes the
    // final logits into this via GPU matvec. The only runtime allocation.
    float* lm_head_scores { nullptr };

    // ── Constructor: load safetensors ─────────────────────────
    Impl(const char* path, SmoeMetalCtx* metal_ctx, const SmoeHeader* vault_hdr,
         const SmoeArchBlock* arch);
    ~Impl();
};

// ── Impl constructor ──────────────────────────────────────────

Scout::Impl::Impl(const char* path, SmoeMetalCtx* metal_ctx, const SmoeHeader* vault_hdr,
                  const SmoeArchBlock* arch) {
    this->metal_ctx = metal_ctx;

    if (vault_hdr) {
        cfg.vocab_size = vault_hdr->vocab_size;
        cfg.d_model = vault_hdr->d_model;
        cfg.ffn_dim = vault_hdr->ffn_dim;
        cfg.num_moe_layers = vault_hdr->num_moe_layers;
        cfg.max_experts_per_layer = vault_hdr->max_experts_per_layer;
        cfg.q2_group_size = vault_hdr->group_size;
        // The topology from vault has num_moe_layers = total layers.
        // For Qwen, has_dense_layer_0 = false, so actual num_moe_layers will be -1.
        // But we will guess has_dense_layer_0 later.
    }

    if (!path) {
        std::fprintf(stderr,
            "[scout] ✗ No weight path provided — engine cannot run without the dense backbone.\n");
        return;
    }

    // ── Open + stat file ──────────────────────────────────────
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr,
            "[scout] ✗ Cannot open '%s': %s\n",
            path, std::strerror(errno));
        return;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        std::fprintf(stderr,
            "[scout] ✗ fstat failed: %s\n",
            std::strerror(errno));
        ::close(fd);
        return;
    }
    mmap_size = static_cast<size_t>(st.st_size);
    info_printf(
        "[scout] Loading scout weights: %s  (%.2f GB)\n",
        path, static_cast<double>(mmap_size) / 1e9);

    // ── mmap (MAP_SHARED — stays in unified DRAM, NOT Direct I/O) ──
    // F_NOCACHE is reserved for the vault streamer.
    // The scout file stays resident — no need to bypass the page cache.
    mmap_base = ::mmap(nullptr, mmap_size,
                       PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);  // fd no longer needed after mmap
    if (mmap_base == MAP_FAILED) {
        mmap_base = nullptr;
        std::fprintf(stderr,
            "[scout] ✗ mmap failed: %s\n",
            std::strerror(errno));
        return;
    }

    const uint8_t* base = static_cast<const uint8_t*>(mmap_base);

    // ── Parse safetensors header ──────────────────────────────
    if (mmap_size < 8) {
        std::fprintf(stderr, "[scout] ✗ File too small.\n");
        ::munmap(mmap_base, mmap_size); mmap_base = nullptr;
        return;
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    // header_len is little-endian; ARM64 macOS is LE, so no byte-swap needed.

    if (header_len > mmap_size - 8) {
        std::fprintf(stderr,
            "[scout] ✗ header_len %llu exceeds file size.\n",
            static_cast<unsigned long long>(header_len));
        ::munmap(mmap_base, mmap_size); mmap_base = nullptr;
        return;
    }

    const char* json_start = reinterpret_cast<const char*>(base + 8);
    const char* json_end   = json_start + header_len;
    // Data region starts immediately after the JSON header
    const uint8_t* data_region = base + 8 + header_len;

    // Prefault the whole mapping in the background: without this the
    // 16 GB of trunk weights are demand-paged in essentially random
    // order during the first prefill (one fault per touched weight row),
    // which serialises against compute. A sequential touch loop streams
    // them in at full NVMe bandwidth instead. One read per 16 KB page.
    ::madvise(mmap_base, mmap_size, MADV_WILLNEED);
    prefault_thread = std::thread([base, sz = mmap_size]() {
        volatile uint8_t sink = 0;
        for (size_t off = 0; off < sz; off += 16384) {
            sink ^= base[off];
        }
        (void)sink;
    });

    info_printf(
        "[scout] Safetensors header: %llu bytes\n",
        static_cast<unsigned long long>(header_len));

    // ── Configure geometry from tensor shapes when no vault header ───
    // (with a vault header the geometry was already taken from it at
    // the top of the constructor).
    if (!vault_hdr) {
        TensorMeta embed_meta = parse_tensor_meta(json_start, json_end, "model.embed_tokens.weight");
        if (embed_meta.valid) {
            cfg.vocab_size = embed_meta.shape[0];
            cfg.d_model = embed_meta.shape[1];
        } else {
            std::fprintf(stderr, "[scout] ✗ Failed to extract model dimensions.\n");
            if (prefault_thread.joinable()) prefault_thread.join(); // reads the mapping
            ::munmap(mmap_base, mmap_size); mmap_base = nullptr;
            return;
        }

        TensorMeta router_meta = parse_tensor_meta(json_start, json_end, "model.layers.1.mlp.gate.weight");
        if (!router_meta.valid) router_meta = parse_tensor_meta(json_start, json_end, "model.layers.0.mlp.gate.weight");
        if (router_meta.valid) cfg.max_experts_per_layer = router_meta.shape[0];
        else cfg.max_experts_per_layer = 64; // Fallback

        cfg.num_moe_layers = 27; // Fallback to DeepSeek hardcode if not passed from Vault yet
    }

    TensorMeta l0_gate_meta = parse_tensor_meta(json_start, json_end, "model.layers.0.mlp.gate_proj.weight");
    if (l0_gate_meta.valid) {
        if (!vault_hdr) cfg.ffn_dim = l0_gate_meta.shape[0];
        cfg.has_dense_layer_0 = true;
    } else {
        cfg.has_dense_layer_0 = false;
        if (!vault_hdr) {
            // No dense L0: read ffn_dim from shared expert if present, else vault header
            TensorMeta shared_meta = parse_tensor_meta(json_start, json_end, "model.layers.0.mlp.shared_expert.gate_proj.weight");
            if (shared_meta.valid) cfg.ffn_dim = shared_meta.shape[0];
            // else: ffn_dim already set from vault_hdr or remains 0 (OK for Qwen3-235B which has no shared expert in scout)
        }
    }

    // Detect Qwen3-style QK-norm (q_norm / k_norm per layer)
    {
        TensorMeta qn = parse_tensor_meta(json_start, json_end, "model.layers.0.self_attn.q_norm.weight");
        cfg.has_qk_norm = qn.valid;
    }

    // Attention projections — needed by both paths (v2 cross-check, v1 inference)
    TensorMeta q_meta = parse_tensor_meta(json_start, json_end, "model.layers.0.self_attn.q_proj.weight");
    TensorMeta k_meta = parse_tensor_meta(json_start, json_end, "model.layers.0.self_attn.k_proj.weight");

    if (arch) {
        // v2 vault: math constants come from the arch block, serialized
        // from the checkpoint's config.json at shatter/upgrade time.
        cfg.rope_theta            = arch->rope_theta;
        cfg.norm_topk_prob        = (arch->flags & SMOE_ARCH_NORM_TOPK) != 0;
        cfg.moe_top_k             = arch->moe_top_k;
        cfg.num_heads             = arch->num_heads;
        cfg.num_kv_heads          = arch->num_kv_heads;
        cfg.head_dim              = arch->head_dim;
        cfg.shared_expert_ffn_dim = arch->shared_expert_ffn_dim;

        // Cross-check the block against tensor evidence — shapes don't lie.
        // Tensor presence stays authoritative for has_qk_norm / has_dense_layer_0.
        const bool arch_qk    = (arch->flags & SMOE_ARCH_QK_NORM) != 0;
        const bool arch_dense = (arch->flags & SMOE_ARCH_DENSE_LAYER_0) != 0;
        if (arch_qk != cfg.has_qk_norm)
            std::fprintf(stderr,
                "[scout] ⚠ arch block says qk_norm=%d but tensor presence says %d — trusting tensors\n",
                (int)arch_qk, (int)cfg.has_qk_norm);
        if (arch_dense != cfg.has_dense_layer_0)
            std::fprintf(stderr,
                "[scout] ⚠ arch block says dense_layer_0=%d but tensor presence says %d — trusting tensors\n",
                (int)arch_dense, (int)cfg.has_dense_layer_0);
        if (q_meta.valid && q_meta.shape[0] != cfg.num_heads * cfg.head_dim)
            std::fprintf(stderr,
                "[scout] ⚠ q_proj rows=%u but arch block implies %u (heads=%u × head_dim=%u)\n",
                q_meta.shape[0], cfg.num_heads * cfg.head_dim, cfg.num_heads, cfg.head_dim);
        if (k_meta.valid && k_meta.shape[0] != cfg.num_kv_heads * cfg.head_dim)
            std::fprintf(stderr,
                "[scout] ⚠ k_proj rows=%u but arch block implies %u (kv_heads=%u × head_dim=%u)\n",
                k_meta.shape[0], cfg.num_kv_heads * cfg.head_dim, cfg.num_kv_heads, cfg.head_dim);
        // The RMS-norm epsilon is compiled into the Metal kernels as 1e-6.
        if (arch->rms_norm_eps < 0.9e-6f || arch->rms_norm_eps > 1.1e-6f)
            std::fprintf(stderr,
                "[scout] ⚠ config rms_norm_eps=%g but kernels hardcode 1e-6\n",
                (double)arch->rms_norm_eps);

        char model_type[sizeof(arch->model_type) + 1] = {};
        std::memcpy(model_type, arch->model_type, sizeof(arch->model_type));
        std::fprintf(stderr,
            "[scout] arch: vault v2 block (%s) — theta=%.0f · top_k=%u · norm_topk=%d · shared_ffn=%u\n",
            model_type, (double)cfg.rope_theta, cfg.moe_top_k,
            (int)cfg.norm_topk_prob, cfg.shared_expert_ffn_dim);
    } else {
        // v1 vault — legacy inference path: Qwen3-family defaults + shapes.
        cfg.rope_theta = 1000000.0f;
        cfg.norm_topk_prob = true;
        cfg.shared_expert_ffn_dim = 0; // overridden if shared tensors found during load

        if (q_meta.valid && k_meta.valid) {
            // Assuming head_dim = 128 (standard for most models including DeepSeek/Qwen)
            cfg.head_dim = 128;
            cfg.num_heads = q_meta.shape[0] / cfg.head_dim;
            cfg.num_kv_heads = k_meta.shape[0] / cfg.head_dim;
        } else {
            // Fallback to MHA defaults (DeepSeek)
            cfg.num_heads = 16;
            cfg.num_kv_heads = 16;
            cfg.head_dim = 128;
        }
        std::fprintf(stderr, "[scout] arch: v1 legacy inference (Qwen3 defaults, theta=1e6)\n");
    }

    std::fprintf(stderr, "[scout] GQA config: heads=%u, kv_heads=%u, head_dim=%u\n", cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);

    // ── Tensor shape expectations ─────────────────────────────
    const size_t EMBED_ELEMS  = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
    const size_t Q_PROJ_ELEMS = static_cast<size_t>(cfg.num_heads * cfg.head_dim) * cfg.d_model;
    const size_t KV_PROJ_ELEMS = static_cast<size_t>(cfg.num_kv_heads * cfg.head_dim) * cfg.d_model;
    const size_t O_PROJ_ELEMS = static_cast<size_t>(cfg.d_model) * (cfg.num_heads * cfg.head_dim);
    const size_t L0_DENSE_ELEMS = static_cast<size_t>(cfg.ffn_dim) * cfg.d_model;

    bool ok = true;

    // The engine's final logits land here via GPU matvec — the Scout's
    // only runtime allocation, made once here.
    lm_head_scores = allocate_aligned_float(cfg.vocab_size);
    if (!lm_head_scores) ok = false;

    // ── Helper: map one tensor into the bf16 mmap region ─────────
    auto load_tensor = [&](const char* name, const uint16_t*& dst, size_t expected_elems) -> bool {
        TensorMeta m = parse_tensor_meta(json_start, json_end, name);
        if (!m.valid) {
            std::fprintf(stderr, "[scout]   ✗ Tensor not found: %s\n", name);
            return false;
        }
        size_t byte_len = m.data_end - m.data_start;
        size_t n_elems  = byte_len / sizeof(uint16_t);
        if (n_elems != expected_elems) {
            std::fprintf(stderr,
                "[scout]   ✗ %s: expected %zu elements, got %zu\n",
                name, expected_elems, n_elems);
            return false;
        }
        dst = reinterpret_cast<const uint16_t*>(data_region + m.data_start);
        return true;
    };

    if (ok) {
        info_printf("[scout] Loading backbone weights for all layers ...\n");
        ok &= load_tensor("model.embed_tokens.weight", w_embed,   EMBED_ELEMS);
        ok &= load_tensor("lm_head.weight",            w_lm_head, EMBED_ELEMS);
        ok &= load_tensor("model.norm.weight",         w_model_norm, cfg.d_model);

        if (cfg.has_dense_layer_0) {
            ok &= load_tensor("model.layers.0.mlp.gate_proj.weight", w_l0_gate, L0_DENSE_ELEMS);
            ok &= load_tensor("model.layers.0.mlp.up_proj.weight", w_l0_up, L0_DENSE_ELEMS);
            ok &= load_tensor("model.layers.0.mlp.down_proj.weight", w_l0_down, L0_DENSE_ELEMS);
        }

        char tensor_name[128];
        uint32_t total_layers = cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0);
    for (uint32_t l = 0; l < total_layers; ++l) {
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.q_proj.weight", l);
            ok &= load_tensor(tensor_name, w_q_proj[l], Q_PROJ_ELEMS);
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.k_proj.weight", l);
            ok &= load_tensor(tensor_name, w_k_proj[l], KV_PROJ_ELEMS);
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.v_proj.weight", l);
            ok &= load_tensor(tensor_name, w_v_proj[l], KV_PROJ_ELEMS);
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.o_proj.weight", l);
            ok &= load_tensor(tensor_name, w_o_proj[l], O_PROJ_ELEMS);
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.input_layernorm.weight", l);
            ok &= load_tensor(tensor_name, w_input_norm[l], cfg.d_model);
            std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.post_attention_layernorm.weight", l);
            ok &= load_tensor(tensor_name, w_post_norm[l], cfg.d_model);

            uint32_t moe_start_layer = cfg.has_dense_layer_0 ? 1 : 0;
            if (l >= moe_start_layer && l < cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0)) {
                std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.mlp.gate.weight", l);
                TensorMeta tm = parse_tensor_meta(json_start, json_end, tensor_name);
                if (!tm.valid) {
                    std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.mlp.gate_proj.weight", l);
                }
                ok &= load_tensor(tensor_name, gate_w[l - moe_start_layer], static_cast<size_t>(cfg.gate_rows()) * cfg.d_model);

                // Shared expert: try both naming conventions; silently skip if absent (Qwen3-235B has none)
                bool found_shared = false;
                const char* shared_variants[][3] = {
                    { "model.layers.%u.mlp.shared_experts.gate_proj.weight",
                      "model.layers.%u.mlp.shared_experts.up_proj.weight",
                      "model.layers.%u.mlp.shared_experts.down_proj.weight" },
                    { "model.layers.%u.mlp.shared_expert.gate_proj.weight",
                      "model.layers.%u.mlp.shared_expert.up_proj.weight",
                      "model.layers.%u.mlp.shared_expert.down_proj.weight" },
                };
                for (auto& variant : shared_variants) {
                    char gn[128], un[128], dn[128];
                    std::snprintf(gn, sizeof(gn), variant[0], l);
                    TensorMeta gm = parse_tensor_meta(json_start, json_end, gn);
                    if (gm.valid) {
                        // Found — load all three and record the dim
                        size_t shared_elems = static_cast<size_t>(gm.shape[0]) * cfg.d_model;
                        std::snprintf(un, sizeof(un), variant[1], l);
                        std::snprintf(dn, sizeof(dn), variant[2], l);
                        ok &= load_tensor(gn, w_shared_gate[l], shared_elems);
                        ok &= load_tensor(un, w_shared_up[l],   shared_elems);
                        ok &= load_tensor(dn, w_shared_down[l], cfg.d_model * gm.shape[0]);
                        if (l == moe_start_layer) {
                            if (arch && cfg.shared_expert_ffn_dim != gm.shape[0])
                                std::fprintf(stderr,
                                    "[scout] ⚠ arch block shared_ffn=%u but tensor shape says %u — trusting tensors\n",
                                    cfg.shared_expert_ffn_dim, gm.shape[0]);
                            cfg.shared_expert_ffn_dim = gm.shape[0];
                        }
                        found_shared = true;
                        break;
                    }
                }
                if (!found_shared) {
                    w_shared_gate[l] = nullptr;
                    w_shared_up[l]   = nullptr;
                    w_shared_down[l] = nullptr;
                    if (l == moe_start_layer && cfg.shared_expert_ffn_dim != 0) {
                        std::fprintf(stderr,
                            "[scout] ⚠ arch block shared_ffn=%u but no shared-expert tensors found — disabling\n",
                            cfg.shared_expert_ffn_dim);
                        cfg.shared_expert_ffn_dim = 0;
                    }
                }

                // Qwen3-specific: Q/K per-head RMS norm weights
                if (cfg.has_qk_norm) {
                    std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.q_norm.weight", l);
                    ok &= load_tensor(tensor_name, w_q_norm[l], cfg.head_dim);
                    std::snprintf(tensor_name, sizeof(tensor_name), "model.layers.%u.self_attn.k_norm.weight", l);
                    ok &= load_tensor(tensor_name, w_k_norm[l], cfg.head_dim);
                }
            }
        }
    }

    if (ok && metal_ctx) {
        info_printf("[scout] Registering buffers with GPU for zero-copy JIT execution ...\n");
        smoe_metal_register_buffer(metal_ctx, w_embed, EMBED_ELEMS * sizeof(uint16_t));
        smoe_metal_register_buffer(metal_ctx, w_lm_head, EMBED_ELEMS * sizeof(uint16_t));
        smoe_metal_register_buffer(metal_ctx, lm_head_scores, cfg.vocab_size * sizeof(float));
    }

    neural_mode = ok;
    if (ok) {
        info_printf(
            "[scout] ✓ Backbone weights ready (D=%u · vocab=%u · %u MoE gates)\n",
            cfg.d_model, cfg.vocab_size, cfg.num_moe_layers);
    } else {
        std::fprintf(stderr, "[scout] ✗ Some tensors failed to load or allocate.\n");
    }
}

// ── Impl destructor ───────────────────────────────────────────

Scout::Impl::~Impl() {
    // The prefault thread reads the mapping — join before munmap.
    if (prefault_thread.joinable()) {
        prefault_thread.join();
    }

    if (mmap_base) {
        ::munmap(mmap_base, mmap_size);
        mmap_base = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════
// §4  Scout public API
// ═══════════════════════════════════════════════════════════════

Scout::Scout(const char* scout_safetensors_path, SmoeMetalCtx* metal_ctx,
             const SmoeHeader* vault_hdr, const SmoeArchBlock* arch)
    : impl_(new Impl(scout_safetensors_path, metal_ctx, vault_hdr, arch))
{}

Scout::~Scout() {
    delete impl_;
}

const SmoeModelConfig& Scout::config() const noexcept {
    return impl_->cfg;
}

const uint16_t* Scout::get_embed() const noexcept { return impl_->w_embed; }
const uint16_t* Scout::get_lm_head() const noexcept { return impl_->w_lm_head; }
const uint16_t* Scout::get_model_norm() const noexcept { return impl_->w_model_norm; }

const uint16_t* Scout::get_l0_gate() const noexcept { return impl_->w_l0_gate; }
const uint16_t* Scout::get_l0_up() const noexcept { return impl_->w_l0_up; }
const uint16_t* Scout::get_l0_down() const noexcept { return impl_->w_l0_down; }

const uint16_t* Scout::get_q_proj(uint32_t l) const noexcept { return impl_->w_q_proj[l]; }
const uint16_t* Scout::get_k_proj(uint32_t l) const noexcept { return impl_->w_k_proj[l]; }
const uint16_t* Scout::get_v_proj(uint32_t l) const noexcept { return impl_->w_v_proj[l]; }
const uint16_t* Scout::get_q_norm(uint32_t layer) const noexcept { return impl_->w_q_norm[layer]; }
const uint16_t* Scout::get_k_norm(uint32_t layer) const noexcept { return impl_->w_k_norm[layer]; }
const uint16_t* Scout::get_o_proj(uint32_t layer) const noexcept { return impl_->w_o_proj[layer]; }

uint32_t Scout::compute_top_k(const float* scores, uint32_t n, uint32_t k, uint32_t* out_indices, float* out_weights, bool norm_topk) const noexcept {
    return top_k(scores, n, k, out_indices, out_weights, norm_topk);
}

const uint16_t* Scout::get_input_norm(uint32_t layer) const noexcept { return impl_->w_input_norm[layer]; }
const uint16_t* Scout::get_post_norm(uint32_t layer) const noexcept { return impl_->w_post_norm[layer]; }
const uint16_t* Scout::get_gate(uint32_t layer) const noexcept { return impl_->gate_w[layer]; }

float* Scout::get_lm_head_scores() const noexcept { return impl_->lm_head_scores; }

const uint16_t* Scout::get_shared_gate(uint32_t layer) const noexcept { return impl_->w_shared_gate[layer]; }
const uint16_t* Scout::get_shared_up(uint32_t l) const noexcept { return impl_->w_shared_up[l]; }
const uint16_t* Scout::get_shared_down(uint32_t l) const noexcept { return impl_->w_shared_down[l]; }

const void* Scout::get_mapped_ptr() const noexcept { return impl_->mmap_base; }
size_t      Scout::get_mapped_size() const noexcept { return impl_->mmap_size; }

} // namespace smoe::scout
