// ═══════════════════════════════════════════════════════════════
// common.hpp — S-MoE Engine · Shared Binary Format Definitions
// ═══════════════════════════════════════════════════════════════
//
// Single source of truth for the .smoe vault binary layout.
// Every struct here is byte-for-byte compatible with the format
// written by scripts/shatter_moe.py.
//
// Constraints (MUST NOT be violated):
//   ① All integers are little-endian.
//   ② All byte offsets are measured from the start of the file.
//   ③ Every expert blob is aligned to PAGE_SIZE (16 KB).
//   ④ Structs are packed (#pragma pack(push,1)) to prevent
//      any compiler-inserted padding from breaking the layout.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace smoe {

// ── Format constants ──────────────────────────────────────────
inline constexpr uint32_t SMOE_VERSION        = 1;
inline constexpr size_t   PAGE_SIZE           = 16'384;   // 16 KB Apple Silicon page
inline constexpr uint32_t Q2_GROUP_SIZE       = 64;       // weights per quantisation group
inline constexpr uint32_t TENSORS_PER_EXPERT  = 3;        // gate_proj, up_proj, down_proj

// Tensor type identifiers (stored in TensorDescriptor::tensor_type)
inline constexpr uint8_t TENSOR_GATE = 0;
inline constexpr uint8_t TENSOR_UP   = 1;
inline constexpr uint8_t TENSOR_DOWN = 2;

// Magic: "SMOE" + 0xDE 0xEA (DeepSeek marker) + 0x00 0x01 (format v1)
inline constexpr std::array<uint8_t, 8> SMOE_MAGIC = {
    'S', 'M', 'O', 'E', 0xDE, 0xEA, 0x00, 0x01
};

[[nodiscard]] inline bool magic_valid(const uint8_t* raw) noexcept {
    return std::memcmp(raw, SMOE_MAGIC.data(), 8) == 0;
}

// ─────────────────────────────────────────────────────────────
// FILE HEADER — 64 bytes, at absolute offset 0
// ─────────────────────────────────────────────────────────────
#pragma pack(push, 1)

struct SmoeHeader {
    uint8_t  magic[8];               // [0]   SMOE_MAGIC
    uint32_t version;                // [8]   SMOE_VERSION
    uint32_t num_moe_layers;         // [12]  number of transformer layers with MoE blocks
    uint32_t max_experts_per_layer;  // [16]  maximum expert count across all MoE layers
    uint32_t total_experts;          // [20]  total entries in the expert table
    uint64_t table_offset;           // [24]  byte offset to the first ExpertEntry
    uint64_t data_offset;            // [32]  byte offset to the first expert blob
    uint32_t group_size;             // [40]  SMOE quantisation group size (= 64)
    uint32_t bits;                   // [44]  quantisation bit depth (2 or 4)
    uint32_t d_model;                // [48]  hidden dimension
    uint32_t vocab_size;             // [52]  number of tokens
    uint32_t ffn_dim;                // [56]  dense/shared FFN intermediate dimension
    uint32_t reserved_ext;           // [60]  padding to 64 bytes
};                                   // [64]

static_assert(sizeof(SmoeHeader) == 64,
    "SmoeHeader layout is broken — must be exactly 64 bytes.");

struct SmoeModelConfig {
    uint32_t d_model;
    uint32_t vocab_size;
    uint32_t ffn_dim;                  // expert intermediate dim (from TensorDescriptor)
    uint32_t shared_expert_ffn_dim {0};// 0 = no shared expert (Qwen3-235B), >0 = DeepSeek
    uint32_t num_moe_layers;
    uint32_t max_experts_per_layer;
    uint32_t q2_group_size;
    bool has_dense_layer_0 {true};     // DeepSeek = true, Qwen3 = false
    bool has_qk_norm {false};          // Qwen3-specific per-head Q/K RMS norm before RoPE
    uint32_t num_heads {16};           // GQA queries (default 16 for backward compat)
    uint32_t num_kv_heads {16};        // GQA keys/values
    uint32_t head_dim {128};           // Attention head dimension
    float    rope_theta {10000.0f};    // RoPE base frequency (10000=DeepSeek, 1000000=Qwen3)
    bool     norm_topk_prob {false};   // Re-normalize routing weights over selected top-k
    // Derived configurations
    uint32_t gate_rows() const { return max_experts_per_layer; }
};

// ─────────────────────────────────────────────────────────────
// EXPERT TABLE ENTRY — 48 bytes, immediately after the header
// N entries, where N = SmoeHeader::total_experts
// ─────────────────────────────────────────────────────────────

struct ExpertEntry {
    uint32_t layer_id;       // [0]  transformer layer index
    uint32_t expert_id;      // [4]  expert index within the layer
    uint64_t byte_offset;    // [8]  absolute file byte offset of this expert's blob
    uint64_t raw_size;       // [16] actual data bytes (before page padding)
    uint64_t padded_size;    // [24] data bytes rounded up to PAGE_SIZE boundary
    uint32_t group_size;     // [32] Q2 group size for this expert (= Q2_GROUP_SIZE)
    uint64_t num_groups;     // [36] total Q2 groups across all three sub-tensors
    uint8_t  reserved[4];   // [44] padding to 48 bytes
};                           // [48]

static_assert(sizeof(ExpertEntry) == 48,
    "ExpertEntry layout is broken — must be exactly 48 bytes.");

// ─────────────────────────────────────────────────────────────
// TENSOR DESCRIPTOR — 44 bytes, 3 per expert (gate, up, down)
// Describes one weight tensor's layout within its parent blob.
// ─────────────────────────────────────────────────────────────

struct TensorDescriptor {
    uint8_t  tensor_type;    // [0]  TENSOR_GATE | TENSOR_UP | TENSOR_DOWN
    uint8_t  ndim;           // [1]  number of dimensions (always 2)
    uint8_t  reserved_0[2]; // [2]  reserved, must be zero
    uint32_t rows;           // [4]  weight matrix rows  (original shape[0])
    uint32_t cols;           // [8]  weight matrix cols  (original shape[1])
    uint64_t packed_offset;  // [12] byte offset of 2-bit packed weights within blob
    uint64_t packed_size;    // [20] byte size of 2-bit packed weights
    uint64_t scales_offset;  // [28] byte offset of float16 scale factors within blob
    uint64_t scales_size;    // [36] byte size of float16 scale factors
};                           // [44]

static_assert(sizeof(TensorDescriptor) == 44,
    "TensorDescriptor layout is broken — must be exactly 44 bytes.");

#pragma pack(pop)

// ─────────────────────────────────────────────────────────────
// SMOE-Q2 DEQUANTISATION — reference formula
// ─────────────────────────────────────────────────────────────
//
// For weight at flat index `i` within a tensor:
//
//   group_idx = i / GROUP_SIZE
//   bit_shift = (i % 4) * 2
//   pack_idx  = i / 4
//
//   scale  = reinterpret_cast<const __fp16*>(scales_ptr)[group_idx]
//   code   = (packed_ptr[pack_idx] >> bit_shift) & 0x3     // ∈ {0,1,2,3}
//   weight = ((float)code - 1.5f) / 1.5f * (float)scale
//
// Code → level mapping:
//   0b00 (0) → −1.000 × scale
//   0b01 (1) → −0.333 × scale
//   0b10 (2) → +0.333 × scale
//   0b11 (3) → +1.000 × scale
//
// This formula is mirrored exactly in src/compute/kernels.metal.
// ─────────────────────────────────────────────────────────────

// Helper: compute the byte offset of the tensor descriptor for
// expert `e` (0-indexed) and sub-tensor `t` (0=gate,1=up,2=down).
[[nodiscard]] inline uint64_t tensor_desc_offset(
    const SmoeHeader& h, uint32_t e, uint32_t t) noexcept
{
    uint64_t table_bytes = static_cast<uint64_t>(h.total_experts) * sizeof(ExpertEntry);
    uint64_t desc_base   = h.table_offset + table_bytes;
    return desc_base
        + static_cast<uint64_t>(e) * TENSORS_PER_EXPERT * sizeof(TensorDescriptor)
        + static_cast<uint64_t>(t) * sizeof(TensorDescriptor);
}

// ── Inline Math Utilities ────────────────────────────────────

inline float* allocate_aligned_float(size_t elems) noexcept {
    size_t bytes = elems * sizeof(float);
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, 16384, bytes) != 0) {
        return nullptr;
    }
    return static_cast<float*>(ptr);
}

inline void free_aligned_float(float* ptr) noexcept {
    if (ptr) {
        ::free(ptr);
    }
}


inline float bf16_to_f32(uint16_t bf) noexcept {
    uint32_t val = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &val, sizeof(f));
    return f;
}

inline void rms_norm_bf16(float* x, const uint16_t* w, uint32_t n) noexcept {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
    ss /= n;
    ss += 1e-6f;
    ss = 1.0f / std::sqrt(ss);
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = (x[i] * ss) * bf16_to_f32(w[i]);
    }
}

#if defined(__ARM_NEON)
// NEON path: bf16 rows are widened in-register (u16 << 16 is exactly
// the bf16→f32 bit pattern — vshll does conversion and load-spread in
// one instruction) and accumulated with fused multiply-add across four
// independent lanes to keep the FMA pipes full. This is the decode
// hot path: the exact-routing gate matvec reads ~94 MB of bf16 per
// generated token; the scalar loop ran it at ~1 GB/s.
//
// NOT bit-exact with the scalar loop: vector lanes reassociate the
// float sum. Near-tie gate scores can therefore flip routing picks —
// verified coherent, not verified identical.
inline void matvec_bf16(float* __restrict__ out,
                   const uint16_t* __restrict__ weight,
                   const float* __restrict__ in,
                   uint32_t rows, uint32_t cols) noexcept {
    for (uint32_t i = 0; i < rows; ++i) {
        const uint16_t* w_row = weight + static_cast<size_t>(i) * cols;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        uint32_t j = 0;
        for (; j + 16 <= cols; j += 16) {
            uint16x8_t h0 = vld1q_u16(w_row + j);
            uint16x8_t h1 = vld1q_u16(w_row + j + 8);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)),  vld1q_f32(in + j));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)),           vld1q_f32(in + j + 4));
            acc2 = vfmaq_f32(acc2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)),  vld1q_f32(in + j + 8));
            acc3 = vfmaq_f32(acc3, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)),           vld1q_f32(in + j + 12));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        for (; j < cols; ++j) sum += bf16_to_f32(w_row[j]) * in[j];
        out[i] = sum;
    }
}
#else
inline void matvec_bf16(float* __restrict__ out,
                   const uint16_t* __restrict__ weight,
                   const float* __restrict__ in,
                   uint32_t rows, uint32_t cols) noexcept {
    for (uint32_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const uint16_t* w_row = weight + i * cols;
        for (uint32_t j = 0; j < cols; ++j) {
            sum += bf16_to_f32(w_row[j]) * in[j];
        }
        out[i] = sum;
    }
}
#endif

// Attention primitives. Clang cannot auto-vectorize float reductions
// without -ffast-math (reassociation), so the scalar loops ran one
// FMA per cycle. head_dim is 128 — the tails never run in practice.
// Same caveat as matvec_bf16: lane reassociation ≠ bit-exact scalar.
#if defined(__ARM_NEON)
inline float dot_f32(const float* __restrict__ a,
                     const float* __restrict__ b, uint32_t n) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline void axpy_f32(float* __restrict__ y, float a,
                     const float* __restrict__ x, uint32_t n) noexcept {
    const float32x4_t va = vdupq_n_f32(a);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     va, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, vld1q_f32(x + i + 4)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
}
#else
inline float dot_f32(const float* __restrict__ a,
                     const float* __restrict__ b, uint32_t n) noexcept {
    float s = 0.0f;
    for (uint32_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline void axpy_f32(float* __restrict__ y, float a,
                     const float* __restrict__ x, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) y[i] += a * x[i];
}
#endif

inline void rms_norm(float* x, const float* w, uint32_t n) noexcept {
    if (!w) return;
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

inline void matvec(float* __restrict__ out,
                   const float* __restrict__ weight,
                   const float* __restrict__ x,
                   uint32_t rows, uint32_t cols) noexcept {
    if (!weight) return;
    for (uint32_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const float* row = weight + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

} // namespace smoe
