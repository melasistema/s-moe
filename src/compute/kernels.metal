// ═══════════════════════════════════════════════════════════════
// kernels.metal — S-MoE Engine · SMOE-Q2 Fused FFN Compute Shader
// ═══════════════════════════════════════════════════════════════
// Phase 3 — Week 3 implementation target.
//
// Will implement:
//   • SMOE-Q2 on-the-fly 2-bit dequantisation in register space.
//   • Fused gate + up projection (element-wise multiply + SiLU gate).
//   • Down projection.
//   • Optimised for Apple Silicon threadgroup memory and SIMD-groups.
//
// Dequantisation formula (mirrors shatter_moe.py exactly):
//   code   = (packed_byte >> ((i % 4) * 2)) & 0x3
//   weight = ((float)code - 1.5h) / 1.5h * scale
// ═══════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// ── Kernel argument buffer layout ────────────────────────────
struct FusedFFNParams {
    uint rows;
    uint cols;
    uint group_size;
};

// ── SiLU activation: x * sigmoid(x) ─────────────────────────
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ── SMOE-Q2 dequantise one weight ────────────────────────────
inline float smoeq2_dequant(
    const device uint8_t*  packed  [[buffer(0)]],
    const device half*     scales  [[buffer(1)]],
    uint weight_idx,
    uint group_size)
{
    uint  pack_idx   = weight_idx / 4;
    uint  bit_shift  = (weight_idx % 4) * 2;
    uint  group_idx  = weight_idx / group_size;

    uint8_t code  = (packed[pack_idx] >> bit_shift) & 0x3;
    float   scale = (float)scales[group_idx];

    // code ∈ {0,1,2,3} → level ∈ {-1.0, -0.333, +0.333, +1.0}
    return ((float)code - 1.5f) / 1.5f * scale;
}

// ── Fused FFN kernel (stub — Week 3) ─────────────────────────
// Gate + Up projections with SiLU gating, then Down projection.
// TODO: full implementation in Week 3.
kernel void smoe_fused_ffn(
    device const uint8_t* gate_packed  [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const uint8_t* up_packed    [[buffer(2)]],
    device const half*    up_scales    [[buffer(3)]],
    device const uint8_t* down_packed  [[buffer(4)]],
    device const half*    down_scales  [[buffer(5)]],
    device const float*   input        [[buffer(6)]],
    device       float*   output       [[buffer(7)]],
    constant FusedFFNParams& params    [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    // TODO — Week 3: tiled GEMV with dequantisation in register space.
    (void)gate_packed; (void)gate_scales;
    (void)up_packed;   (void)up_scales;
    (void)down_packed; (void)down_scales;
    (void)input;       (void)params;
    output[gid.x] = 0.0f;
}
