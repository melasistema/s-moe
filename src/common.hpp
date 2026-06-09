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
    uint32_t group_size;             // [40]  SMOE-Q2 quantisation group size (= 64)
    uint8_t  reserved[20];           // [44]  must be zero
};                                   // [64]

static_assert(sizeof(SmoeHeader) == 64,
    "SmoeHeader layout is broken — must be exactly 64 bytes.");

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

} // namespace smoe
