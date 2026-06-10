// ═══════════════════════════════════════════════════════════════
// scout.cpp — S-MoE Engine · Surface Scout (Week 5: Neural)
// ═══════════════════════════════════════════════════════════════
// Phase 5 — Week 5
//
// Architecture: SAR-phonon analogy
// ──────────────────────────────────────────────────────────────
// The Surface Scout is a lightweight distilled model (~1.5B params)
// that runs two heads simultaneously:
//
//   ① Language head — standard autoregressive next-token prediction
//      via a final linear projection (lm_head) over the hidden state.
//
//   ② Routing head — for each MoE layer L in [1..27], a small gate
//      projection (shape [64, 2048]) predicts which 8 of 64 experts
//      the heavy frontier model will activate K steps ahead.
//      This decouples routing-matrix evaluation from the current
//      execution step — the phonon analogy: we measure the *echo*
//      of the routing decision before it's needed by the GPU.
//
// Week 5 neural pass (single token, no KV-cache):
//   1. Embed:          hidden = embed_tokens[token_id]        (2048,)
//   2. Attention-1:    run layer-1 self-attention on hidden
//      a. RMS norm    (input_layernorm.weight)
//      b. Q/K/V proj  (q/k/v_proj.weight)
//      c. Scaled dot-product attention over a 64-token ring cache
//      d. O proj      (o_proj.weight)  + residual
//      e. Post-attn RMS norm
//   3. Gate routing:   for L in [1..27]:
//                         scores = hidden @ gate[L].T → softmax → top-8
//   4. Next token:     lm_head_scores = hidden @ lm_head.T → argmax
//
// Weight format: safetensors (parsed from scratch, no external libs)
//   • 8-byte little-endian uint64 header_len
//   • header_len bytes of UTF-8 JSON metadata
//   • raw bf16 data (data_offsets relative to end of JSON)
//
// Memory invariants (GEMINI.md):
//   ① Zero heap allocations inside forward() — all buffers are
//     members of Impl, pre-allocated in the constructor.
//   ② Synchronisation is via std::atomic only — no OS mutexes.
//   ③ The safetensors file is mmap()‐ed in MAP_PRIVATE mode and
//     stays resident (scout stays in DRAM). The F_NOCACHE Direct
//     I/O flag is NOT applied here — that is reserved for the
//     vault streamer which must bypass the page cache.
//
// Fallback:
//   When scout_safetensors_path == nullptr (or file cannot be
//   opened), the implementation reverts to the Week 4 heuristic
//   triple-strategy oracle (n-gram / recency / uniform).
// ═══════════════════════════════════════════════════════════════

#include "scout.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// POSIX / macOS headers for mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace smoe::scout {

// ═══════════════════════════════════════════════════════════════
// §1  Compile-time constants
// ═══════════════════════════════════════════════════════════════

// ── Model dimensions ──────────────────────────────────────────
inline constexpr uint32_t D_MODEL        = 2048;    // hidden dimension
inline constexpr uint32_t VOCAB_SIZE     = 102400;  // lm_head / embed rows
inline constexpr uint32_t NUM_MOE_LAYERS = 27;      // layers 1..27 have gate
inline constexpr uint32_t GATE_ROWS      = 64;      // experts per gate

// Context ring depth for the single-layer attention
inline constexpr uint32_t ATTN_CTX       = 64;

// ── Heuristic constants (fallback path, unchanged from Week 4) ─
inline constexpr uint32_t NGRAM_WINDOW    = 4;
inline constexpr uint32_t NGRAM_TABLE     = 4096;
inline constexpr uint32_t N_HIST          = MAX_ACTIVE;
inline constexpr uint32_t EXPERT_UNIVERSE = 256;
inline constexpr uint32_t MAX_LAYERS      = 64;
inline constexpr uint32_t RECENCY_DELTA   = 3;
inline constexpr uint32_t HISTORY_CAP     = 1024;

// ═══════════════════════════════════════════════════════════════
// §2  Bfloat16 conversion
// ═══════════════════════════════════════════════════════════════

// bf16 is the top 16 bits of IEEE-754 float32.
// Conversion: sign-extend the 16-bit value into the upper word.
static inline float bf16_to_f32(uint16_t bf) noexcept {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert a packed bf16 array to float32 in-place into dst.
static void convert_bf16_block(float* dst, const uint16_t* src, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_f32(src[i]);
    }
}

// ═══════════════════════════════════════════════════════════════
// §3  Math kernels (zero allocation, no external BLAS)
// ═══════════════════════════════════════════════════════════════

// RMS LayerNorm: out[i] = x[i] / rms(x) * weight[i]
// Applied in-place.
static void rms_norm(float* x, const float* w, uint32_t n) noexcept {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(n) + 1e-6f);
    float inv_rms = 1.0f / rms;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = x[i] * inv_rms * w[i];
    }
}

// Matrix-vector multiply: out[i] = dot(weight[i*cols .. i*cols+cols-1], x)
// weight is row-major [rows × cols].
static void matvec(float* __restrict__ out,
                   const float* __restrict__ weight,
                   const float* __restrict__ x,
                   uint32_t rows, uint32_t cols) noexcept {
    for (uint32_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const float* row = weight + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

// In-place softmax over n elements.
static void softmax(float* x, uint32_t n) noexcept {
    float max_val = x[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum  += x[i];
    }
    float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

// Select top-k indices from scores[n] into out_indices[k].
// Returns k (or fewer if n < k).
// Uses a simple partial selection — fine for k=8, n=64.
static uint32_t top_k(const float* scores, uint32_t n, uint32_t k,
                       uint32_t* out_indices) noexcept {
    uint32_t count = std::min(k, n);
    // Fill with sentinel -1 (unused)
    static_assert(sizeof(uint32_t) == 4);
    // We mark selected indices to avoid duplicates
    // Use a small stack-local copy of scores — n ≤ 64, fine on stack
    float   tmp[GATE_ROWS];
    uint32_t idx[GATE_ROWS];
    for (uint32_t i = 0; i < n; ++i) { tmp[i] = scores[i]; idx[i] = i; }
    for (uint32_t s = 0; s < count; ++s) {
        uint32_t best = s;
        for (uint32_t i = s + 1; i < n; ++i) {
            if (tmp[i] > tmp[best]) best = i;
        }
        // Swap
        float   ft = tmp[s]; tmp[s] = tmp[best]; tmp[best] = ft;
        uint32_t it = idx[s]; idx[s] = idx[best]; idx[best] = it;
        out_indices[s] = idx[s];
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════
// §4  Minimal safetensors header parser
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
// §5  Weight tensor pointers (into mmap)
// ═══════════════════════════════════════════════════════════════

// All weight arrays as pointers into the mmap region.
// The bf16 data is converted to float32 at construction time and
// stored in pre-allocated Impl member arrays.

// ═══════════════════════════════════════════════════════════════
// §6  Heuristic data structures (Week 4 fallback, unchanged)
// ═══════════════════════════════════════════════════════════════

struct NgramBucket {
    uint32_t experts[MAX_LAYERS][N_HIST] {};
    uint32_t counts [MAX_LAYERS][N_HIST] {};
    uint32_t head   [MAX_LAYERS]         {};
};

struct LayerRecency {
    uint32_t last_experts[MAX_ACTIVE] {};
    uint32_t count { 0 };
};

// ═══════════════════════════════════════════════════════════════
// §7  Scout::Impl
// ═══════════════════════════════════════════════════════════════

struct Scout::Impl {

    // ── Mode flag ──────────────────────────────────────────────
    bool neural_mode { false };   // true ↔ safetensors loaded

    // ── mmap state ────────────────────────────────────────────
    void*    mmap_base { nullptr };
    size_t   mmap_size { 0 };

    // ── Neural weight storage (float32, pre-allocated) ────────
    // Each array is large enough for the tensor it represents.
    // All allocated as flat C arrays — no std::vector, no new[].

    // Embedding table: [VOCAB_SIZE × D_MODEL]
    // ~800 MB for 102400 × 2048 × 4 bytes — huge but fits in 16 GB UMA.
    // We use a pointer into a heap allocation done ONCE at constructor time.
    float* w_embed      { nullptr };   // [VOCAB_SIZE × D_MODEL]
    float* w_lm_head    { nullptr };   // [VOCAB_SIZE × D_MODEL]

    // Layer 1 attention weights: [D_MODEL × D_MODEL]
    float* w_q_proj     { nullptr };   // [D_MODEL × D_MODEL]
    float* w_k_proj     { nullptr };   // [D_MODEL × D_MODEL]
    float* w_v_proj     { nullptr };   // [D_MODEL × D_MODEL]
    float* w_o_proj     { nullptr };   // [D_MODEL × D_MODEL]

    // Layer 1 norms: [D_MODEL]
    float  w_input_norm [D_MODEL]     {};
    float  w_post_norm  [D_MODEL]     {};

    // Final norm: [D_MODEL]
    float  w_model_norm [D_MODEL]     {};

    // Gate weights for MoE layers 1..27: [NUM_MOE_LAYERS × GATE_ROWS × D_MODEL]
    // gate[l] points to gate_weights_storage + l*GATE_ROWS*D_MODEL
    static constexpr size_t GATE_STORAGE_ELEMS =
        static_cast<size_t>(NUM_MOE_LAYERS) * GATE_ROWS * D_MODEL;
    float* gate_weights_storage { nullptr };   // flat [27 × 64 × 2048]
    // Convenience: gate_w[l] = row-major [GATE_ROWS × D_MODEL] for MoE layer l+1
    // (index 0 = layer 1, ..., index 26 = layer 27)
    float* gate_w[NUM_MOE_LAYERS] {};

    // ── Pre-allocated compute buffers (forward(), zero alloc) ─
    float hidden    [D_MODEL]  {};       // current hidden state
    float normed    [D_MODEL]  {};       // scratch for post-norm
    float qbuf      [D_MODEL]  {};
    float kbuf      [D_MODEL]  {};
    float vbuf      [D_MODEL]  {};
    float attn_out  [D_MODEL]  {};

    // Attention context ring: last ATTN_CTX K-vectors [ATTN_CTX × D_MODEL]
    float k_cache   [ATTN_CTX][D_MODEL] {};
    float v_cache   [ATTN_CTX][D_MODEL] {};
    uint32_t ctx_pos  { 0 };   // write head into ring
    uint32_t ctx_fill { 0 };   // number of valid entries

    // Attention score buffer [ATTN_CTX]
    float attn_scores[ATTN_CTX] {};

    // Gate score buffer [GATE_ROWS]
    float gate_scores[GATE_ROWS] {};

    // LM head score buffer — we do NOT pre-allocate 102400 floats on stack.
    // Instead we argmax inline over the bf16 row of lm_head to avoid the
    // large buffer.  We keep a tiny scratch for the best-score scan.
    // (This is still zero-alloc — we scan the converted float32 weight array.)

    // ── Heuristic fallback state (Week 4, preserved) ──────────
    NgramBucket ngram_table[NGRAM_TABLE] {};
    LayerRecency recency   [MAX_LAYERS]  {};
    uint32_t context_ring  [NGRAM_WINDOW] {};
    uint32_t context_pos_h { 0 };
    uint32_t context_fill_h{ 0 };
    uint32_t history       [HISTORY_CAP] {};
    uint32_t history_pos   { 0 };
    uint32_t history_fill  { 0 };
    static constexpr uint32_t FREQ_TABLE_SIZE = 65536;
    uint32_t token_freq    [FREQ_TABLE_SIZE] {};
    uint32_t num_moe_layers    { 21 };
    uint32_t experts_per_layer { EXPERT_UNIVERSE };
    uint64_t step              { 0 };

    // ── Constructor: load safetensors ─────────────────────────
    explicit Impl(const char* path);
    ~Impl();

    // ── Helpers: neural path ──────────────────────────────────
    // Run the full neural forward pass.  NO allocations here.
    ScoutOutput neural_forward(uint32_t token_id) noexcept;

    // ── Helpers: heuristic path (Week 4) ─────────────────────
    uint32_t ngram_hash() const noexcept;
    void     observe(uint32_t layer_id, uint32_t expert_id) noexcept;
    uint32_t predict(uint32_t layer_id, uint32_t* out) const noexcept;
    uint32_t next_token_heuristic(uint32_t current_token) const noexcept;
    void     push_token(uint32_t token_id) noexcept;
    ScoutOutput heuristic_forward(uint32_t token_id) noexcept;
};

// ── Impl constructor ──────────────────────────────────────────

Scout::Impl::Impl(const char* path) {
    if (!path) {
        std::fprintf(stderr,
            "[scout] Heuristic mode — no weight path provided.\n");
        return;
    }

    // ── Open + stat file ──────────────────────────────────────
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr,
            "[scout] ✗ Cannot open '%s': %s\n"
            "[scout]   Falling back to heuristic mode.\n",
            path, std::strerror(errno));
        return;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        std::fprintf(stderr,
            "[scout] ✗ fstat failed: %s — heuristic fallback.\n",
            std::strerror(errno));
        ::close(fd);
        return;
    }
    mmap_size = static_cast<size_t>(st.st_size);
    std::fprintf(stderr,
        "[scout] Loading scout weights: %s  (%.2f GB)\n",
        path, static_cast<double>(mmap_size) / 1e9);

    // ── mmap (MAP_PRIVATE — stays in unified DRAM, NOT Direct I/O) ──
    // F_NOCACHE is reserved for the vault streamer.
    // The scout file stays resident — no need to bypass the page cache.
    mmap_base = ::mmap(nullptr, mmap_size,
                       PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // fd no longer needed after mmap
    if (mmap_base == MAP_FAILED) {
        mmap_base = nullptr;
        std::fprintf(stderr,
            "[scout] ✗ mmap failed: %s — heuristic fallback.\n",
            std::strerror(errno));
        return;
    }

    const uint8_t* base = static_cast<const uint8_t*>(mmap_base);

    // ── Parse safetensors header ──────────────────────────────
    if (mmap_size < 8) {
        std::fprintf(stderr, "[scout] ✗ File too small — heuristic fallback.\n");
        ::munmap(mmap_base, mmap_size); mmap_base = nullptr;
        return;
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    // header_len is little-endian; ARM64 macOS is LE, so no byte-swap needed.

    if (header_len > mmap_size - 8) {
        std::fprintf(stderr,
            "[scout] ✗ header_len %llu exceeds file size — heuristic fallback.\n",
            static_cast<unsigned long long>(header_len));
        ::munmap(mmap_base, mmap_size); mmap_base = nullptr;
        return;
    }

    const char* json_start = reinterpret_cast<const char*>(base + 8);
    const char* json_end   = json_start + header_len;
    // Data region starts immediately after the JSON header
    const uint8_t* data_region = base + 8 + header_len;

    std::fprintf(stderr,
        "[scout] Safetensors header: %llu bytes\n",
        static_cast<unsigned long long>(header_len));

    // ── Allocate float32 weight arrays (ONE-TIME, in constructor) ──
    // Large allocations done here, never inside forward().
    static constexpr size_t EMBED_ELEMS  = static_cast<size_t>(VOCAB_SIZE) * D_MODEL;
    static constexpr size_t ATTN_W_ELEMS = static_cast<size_t>(D_MODEL)    * D_MODEL;

    w_embed              = new float[EMBED_ELEMS];
    w_lm_head            = new float[EMBED_ELEMS];
    w_q_proj             = new float[ATTN_W_ELEMS];
    w_k_proj             = new float[ATTN_W_ELEMS];
    w_v_proj             = new float[ATTN_W_ELEMS];
    w_o_proj             = new float[ATTN_W_ELEMS];
    gate_weights_storage = new float[GATE_STORAGE_ELEMS];

    // Wire up gate_w convenience pointers
    for (uint32_t l = 0; l < NUM_MOE_LAYERS; ++l) {
        gate_w[l] = gate_weights_storage +
                    static_cast<size_t>(l) * GATE_ROWS * D_MODEL;
    }

    // ── Helper: load one tensor into a float32 buffer ─────────
    auto load_tensor = [&](const char* name, float* dst, size_t expected_elems) -> bool {
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
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            data_region + m.data_start);
        convert_bf16_block(dst, src, n_elems);
        return true;
    };

    // ── Load embedding and lm_head ────────────────────────────
    std::fprintf(stderr, "[scout] Loading embeddings ...\n");
    bool ok = true;
    ok &= load_tensor("model.embed_tokens.weight", w_embed,   EMBED_ELEMS);
    ok &= load_tensor("lm_head.weight",            w_lm_head, EMBED_ELEMS);

    // ── Load layer-1 attention weights ────────────────────────
    std::fprintf(stderr, "[scout] Loading layer 1 attention ...\n");
    ok &= load_tensor("model.layers.1.self_attn.q_proj.weight", w_q_proj, ATTN_W_ELEMS);
    ok &= load_tensor("model.layers.1.self_attn.k_proj.weight", w_k_proj, ATTN_W_ELEMS);
    ok &= load_tensor("model.layers.1.self_attn.v_proj.weight", w_v_proj, ATTN_W_ELEMS);
    ok &= load_tensor("model.layers.1.self_attn.o_proj.weight", w_o_proj, ATTN_W_ELEMS);

    // ── Load layer-1 norms ────────────────────────────────────
    ok &= load_tensor("model.layers.1.input_layernorm.weight",            w_input_norm, D_MODEL);
    ok &= load_tensor("model.layers.1.post_attention_layernorm.weight",   w_post_norm,  D_MODEL);

    // ── Load final norm ───────────────────────────────────────
    ok &= load_tensor("model.norm.weight", w_model_norm, D_MODEL);

    // ── Load MoE gate weights for layers 1..27 ────────────────
    std::fprintf(stderr, "[scout] Loading MoE gate weights (layers 1–27) ...\n");
    char tensor_name[128];
    for (uint32_t l = 1; l <= NUM_MOE_LAYERS; ++l) {
        std::snprintf(tensor_name, sizeof(tensor_name),
                      "model.layers.%u.mlp.gate.weight", l);
        ok &= load_tensor(tensor_name, gate_w[l - 1],
                          static_cast<size_t>(GATE_ROWS) * D_MODEL);
    }

    if (!ok) {
        std::fprintf(stderr,
            "[scout] ⚠  Some tensors failed to load — falling back to heuristic.\n");
        // Free allocations; heuristic path will run
        delete[] w_embed;              w_embed              = nullptr;
        delete[] w_lm_head;            w_lm_head            = nullptr;
        delete[] w_q_proj;             w_q_proj             = nullptr;
        delete[] w_k_proj;             w_k_proj             = nullptr;
        delete[] w_v_proj;             w_v_proj             = nullptr;
        delete[] w_o_proj;             w_o_proj             = nullptr;
        delete[] gate_weights_storage; gate_weights_storage = nullptr;
        ::munmap(mmap_base, mmap_size);
        mmap_base = nullptr;
        return;
    }

    neural_mode = true;
    std::fprintf(stderr,
        "[scout] ✓ Neural scout ready (Week 5 · D=%u · vocab=%u · %u MoE gates)\n",
        D_MODEL, VOCAB_SIZE, NUM_MOE_LAYERS);
}

// ── Impl destructor ───────────────────────────────────────────

Scout::Impl::~Impl() {
    delete[] w_embed;
    delete[] w_lm_head;
    delete[] w_q_proj;
    delete[] w_k_proj;
    delete[] w_v_proj;
    delete[] w_o_proj;
    delete[] gate_weights_storage;

    if (mmap_base) {
        ::munmap(mmap_base, mmap_size);
        mmap_base = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════
// §8  Neural forward pass (zero allocation)
// ═══════════════════════════════════════════════════════════════

ScoutOutput Scout::Impl::neural_forward(uint32_t token_id) noexcept {
    ScoutOutput out {};

    // ── 1. Token embedding lookup ─────────────────────────────
    // hidden = embed_tokens[token_id], clamped to valid range.
    const uint32_t safe_id = (token_id < VOCAB_SIZE) ? token_id : 0;
    const float*   emb_row = w_embed + static_cast<size_t>(safe_id) * D_MODEL;
    std::memcpy(hidden, emb_row, D_MODEL * sizeof(float));

    // ── 2. Layer-1 attention (stateless per-token, tiny KV ring) ──

    // a. Input RMS LayerNorm — operates on a copy (normed[])
    std::memcpy(normed, hidden, D_MODEL * sizeof(float));
    rms_norm(normed, w_input_norm, D_MODEL);

    // b. Q/K/V projections
    matvec(qbuf, w_q_proj, normed, D_MODEL, D_MODEL);
    matvec(kbuf, w_k_proj, normed, D_MODEL, D_MODEL);
    matvec(vbuf, w_v_proj, normed, D_MODEL, D_MODEL);

    // c. Write K/V into ring cache
    const uint32_t slot = ctx_pos % ATTN_CTX;
    std::memcpy(k_cache[slot], kbuf, D_MODEL * sizeof(float));
    std::memcpy(v_cache[slot], vbuf, D_MODEL * sizeof(float));
    ctx_pos  = (ctx_pos + 1) % ATTN_CTX;
    if (ctx_fill < ATTN_CTX) ++ctx_fill;

    // d. Scaled dot-product attention: scores[i] = dot(Q, K[i]) / sqrt(D)
    const float scale = 1.0f / std::sqrt(static_cast<float>(D_MODEL));
    const uint32_t valid = ctx_fill;
    for (uint32_t i = 0; i < valid; ++i) {
        // K ring: most recent is at (ctx_pos - 1 + ATTN_CTX) % ATTN_CTX
        uint32_t ki = (ctx_pos - 1 - i + 2 * ATTN_CTX) % ATTN_CTX;
        float dot_qk = 0.0f;
        const float* krow = k_cache[ki];
        for (uint32_t d = 0; d < D_MODEL; ++d) {
            dot_qk += qbuf[d] * krow[d];
        }
        attn_scores[i] = dot_qk * scale;
    }
    // Softmax over valid positions
    softmax(attn_scores, valid);

    // Weighted sum of V vectors → attn_out
    std::memset(attn_out, 0, D_MODEL * sizeof(float));
    for (uint32_t i = 0; i < valid; ++i) {
        uint32_t vi = (ctx_pos - 1 - i + 2 * ATTN_CTX) % ATTN_CTX;
        const float alpha = attn_scores[i];
        const float* vrow = v_cache[vi];
        for (uint32_t d = 0; d < D_MODEL; ++d) {
            attn_out[d] += alpha * vrow[d];
        }
    }

    // e. Output projection + residual
    // normed (scratch) ← o_proj @ attn_out
    matvec(normed, w_o_proj, attn_out, D_MODEL, D_MODEL);
    // residual: hidden += normed
    for (uint32_t d = 0; d < D_MODEL; ++d) {
        hidden[d] += normed[d];
    }

    // f. Post-attention RMS norm (in-place on hidden via normed copy)
    std::memcpy(normed, hidden, D_MODEL * sizeof(float));
    rms_norm(normed, w_post_norm, D_MODEL);
    // We carry normed as the "post-attn hidden" for routing.
    // For simplicity (Week 5, single attention layer): normed → hidden.
    std::memcpy(hidden, normed, D_MODEL * sizeof(float));

    // ── 3. Final model norm ───────────────────────────────────
    // Apply model.norm before lm_head and gate projections.
    rms_norm(hidden, w_model_norm, D_MODEL);

    // ── 4. Gate routing for MoE layers 1..27 ─────────────────
    // For each MoE layer: gate_scores = hidden @ gate_w[l].T → softmax → top-8.
    // We populate LOOKAHEAD_K steps by cycling through layers.
    // Each lookahead slot predicts one MoE layer's expert activations.
    for (uint32_t k = 0; k < LOOKAHEAD_K; ++k) {
        // Cycle through layers 0..26 (= model layers 1..27)
        const uint32_t li = k % NUM_MOE_LAYERS;
        ExpertPrediction& pred = out.lookahead[k];
        pred.layer_id = li + 1;  // model layer index (1-based)

        // gate_scores[e] = dot(hidden, gate_w[li][e*D_MODEL .. +D_MODEL])
        matvec(gate_scores, gate_w[li], hidden, GATE_ROWS, D_MODEL);
        softmax(gate_scores, GATE_ROWS);

        // top-8 expert indices
        pred.count = top_k(gate_scores, GATE_ROWS, MAX_ACTIVE, pred.expert_ids);
    }

    // ── 5. Next token: argmax over lm_head scores ────────────
    // We scan row-by-row to avoid allocating a VOCAB_SIZE float buffer.
    // lm_head.weight is [VOCAB_SIZE × D_MODEL]; we compute dot(hidden, row_v)
    // for each row v and track the running argmax.
    {
        uint32_t best_tok   = 0;
        float    best_score = -1e38f;
        for (uint32_t v = 0; v < VOCAB_SIZE; ++v) {
            const float* row = w_lm_head + static_cast<size_t>(v) * D_MODEL;
            float score = 0.0f;
            for (uint32_t d = 0; d < D_MODEL; ++d) {
                score += row[d] * hidden[d];
            }
            if (score > best_score) {
                best_score = score;
                best_tok   = v;
            }
        }
        out.next_token_id = best_tok;
    }

    ++step;
    return out;
}

// ═══════════════════════════════════════════════════════════════
// §9  Heuristic path (Week 4 — preserved verbatim)
// ═══════════════════════════════════════════════════════════════

uint32_t Scout::Impl::ngram_hash() const noexcept {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < NGRAM_WINDOW; ++i) {
        uint32_t tok = context_ring[(context_pos_h + i) % NGRAM_WINDOW];
        h ^= tok;
        h *= 16777619u;
    }
    return h & (NGRAM_TABLE - 1);
}

void Scout::Impl::observe(uint32_t layer_id, uint32_t expert_id) noexcept {
    if (layer_id >= MAX_LAYERS) return;
    NgramBucket& bucket = ngram_table[ngram_hash()];
    uint32_t     slot   = bucket.head[layer_id] % N_HIST;
    bucket.experts[layer_id][slot] = expert_id;
    bucket.counts [layer_id][slot]++;
    bucket.head   [layer_id] = slot + 1;

    LayerRecency& rec = recency[layer_id];
    if (rec.count < MAX_ACTIVE) {
        rec.last_experts[rec.count++] = expert_id;
    } else {
        rec.last_experts[step % MAX_ACTIVE] = expert_id;
    }
}

uint32_t Scout::Impl::predict(uint32_t layer_id, uint32_t* out) const noexcept {
    if (layer_id >= MAX_LAYERS) return 0;

    uint32_t n = 0;
    bool     seen[EXPERT_UNIVERSE] {};

    // Strategy 1: n-gram bucket
    const NgramBucket& bucket = ngram_table[ngram_hash()];
    for (uint32_t i = 0; i < N_HIST && n < MAX_ACTIVE; ++i) {
        uint32_t e = bucket.experts[layer_id][i];
        if (e < experts_per_layer && !seen[e]) {
            out[n++] = e; seen[e] = true;
        }
    }

    // Strategy 2: recency neighbourhood
    const LayerRecency& rec = recency[layer_id];
    for (uint32_t r = 0; r < rec.count && n < MAX_ACTIVE; ++r) {
        uint32_t center = rec.last_experts[r];
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

    // Strategy 3: uniform fallback
    if (n < MAX_ACTIVE && experts_per_layer > 0) {
        uint32_t step_size = experts_per_layer / MAX_ACTIVE;
        if (step_size == 0) step_size = 1;
        for (uint32_t i = 0; i < experts_per_layer && n < MAX_ACTIVE; i += step_size) {
            if (!seen[i]) { out[n++] = i; seen[i] = true; }
        }
    }

    return n;
}

uint32_t Scout::Impl::next_token_heuristic(uint32_t current_token) const noexcept {
    uint32_t best_tok   = (current_token + 1) % FREQ_TABLE_SIZE;
    uint32_t best_count = 0;
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

void Scout::Impl::push_token(uint32_t token_id) noexcept {
    context_ring[context_pos_h % NGRAM_WINDOW] = token_id;
    context_pos_h  = (context_pos_h + 1) % NGRAM_WINDOW;
    context_fill_h = std::min(context_fill_h + 1, NGRAM_WINDOW);

    history[history_pos] = token_id;
    history_pos          = (history_pos + 1) % HISTORY_CAP;
    history_fill         = std::min(history_fill + 1, HISTORY_CAP);

    if (token_id < FREQ_TABLE_SIZE) token_freq[token_id]++;
}

ScoutOutput Scout::Impl::heuristic_forward(uint32_t token_id) noexcept {
    ScoutOutput out {};
    push_token(token_id);
    out.next_token_id = next_token_heuristic(token_id);
    for (uint32_t k = 0; k < LOOKAHEAD_K; ++k) {
        ExpertPrediction& pred = out.lookahead[k];
        pred.layer_id = k % num_moe_layers;
        pred.count    = predict(pred.layer_id, pred.expert_ids);
    }
    ++step;
    return out;
}

// ═══════════════════════════════════════════════════════════════
// §10  Scout public API
// ═══════════════════════════════════════════════════════════════

Scout::Scout(const char* scout_safetensors_path)
    : impl_(new Impl(scout_safetensors_path))
{}

Scout::~Scout() {
    delete impl_;
}

ScoutOutput Scout::forward(uint32_t token_id) {
    if (impl_->neural_mode) {
        return impl_->neural_forward(token_id);
    }
    return impl_->heuristic_forward(token_id);
}

void Scout::reset_context() {
    // Neural: reset KV-cache ring
    impl_->ctx_pos  = 0;
    impl_->ctx_fill = 0;
    std::memset(impl_->k_cache, 0, sizeof(impl_->k_cache));
    std::memset(impl_->v_cache, 0, sizeof(impl_->v_cache));

    // Heuristic: reset context ring + history (preserve learned freq tables)
    std::memset(impl_->context_ring, 0, sizeof(impl_->context_ring));
    impl_->context_pos_h  = 0;
    impl_->context_fill_h = 0;

    std::memset(impl_->history, 0, sizeof(impl_->history));
    impl_->history_pos  = 0;
    impl_->history_fill = 0;

    for (auto& rec : impl_->recency) {
        rec = LayerRecency{};
    }

    impl_->step = 0;
    std::fprintf(stderr, "[scout] Context reset.\n");
}

} // namespace smoe::scout
