// ═══════════════════════════════════════════════════════════════
// kernels.metal — S-MoE Engine · SMOE-Q2 Fused FFN Compute Shader
// ═══════════════════════════════════════════════════════════════
// Phase 3 — Week 3
//
// Implements a GEMV-style fused kernel for one expert FFN block:
//   1. Gate projection:  gate_out[i] = dot(gate_row_i, x)
//   2. Up projection:    up_out[i]   = dot(up_row_i,   x)
//   3. SiLU gating:      hidden[i]   = silu(gate_out[i]) * up_out[i]
//   4. Down projection:  out[j]      = dot(down_col_j, hidden)
//
// All weight access is on-the-fly 2-bit SMOE-Q2 dequantisation in
// register space. No decompressed weight matrix is ever materialised.
//
// Threading model:
//   Grid dim:  (rows,)        — one thread per output element
//   Tgroup:    TGROUP_SIZE    — threads share the input vector via
//                               threadgroup memory to avoid redundant
//                               global reads.
//
// Design invariants:
//   ① Input vector is broadcast-loaded into threadgroup SRAM once.
//   ② Dequantisation is purely register-local — no scratch buffers.
//   ③ Half-precision accumulation used for inner dot-products;
//      promoted to float32 before SiLU to preserve gate accuracy.
//   ④ Loop is unrolled x4 to exploit the natural 4-codes-per-byte
//      packing: one byte loaded → four weights decoded immediately.
// ═══════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

// ── Tuning constants ─────────────────────────────────────────
// 256 threads per threadgroup: on Apple M-series GPU a 256-thread
// occupancy point typically saturates the SIMD width (32) × 8 waves.
constant uint TGROUP_SIZE = 256;

// ── Kernel argument buffer layout ────────────────────────────
struct FusedFFNParams {
    uint rows;        // weight matrix rows  (= intermediate / hidden dim)
    uint cols;        // weight matrix cols  (= input dim)
    uint group_size;  // SMOE quantisation group size (= 64)
    uint bits;        // bits per weight (2 or 4)
};

// ── SiLU activation: x · σ(x) ───────────────────────────────
inline float silu(float x) {
    return x * (1.0f / (1.0f + exp(-x)));
}

// ── SMOE-Q2: decode one weight at flat index `wi` ────────────
//
//   pack_idx  = wi / 4           — which byte holds this code
//   bit_shift = (wi % 4) * 2    — which 2-bit slot within that byte
//   group_idx = wi / group_size  — which fp16 scale applies
//
//   code  ∈ {0,1,2,3}  →  level ∈ {-1, -0.333, +0.333, +1} × scale
//
inline float smoeq2_dequant(
    device const uint8_t* packed,
    device const half*    scales,
    uint wi,
    uint group_size)
{
    uint    pack_idx  = wi >> 2;                        // wi / 4
    uint    bit_shift = (wi & 3u) << 1u;               // (wi % 4) * 2
    uint    group_idx = wi / group_size;

    uint8_t code  = (packed[pack_idx] >> bit_shift) & 0x3u;
    float   scale = float(scales[group_idx]);

    // Affine mapping: centre 0–3 around zero, then scale
    return ((float(code) - 1.5f) * (1.0f / 1.5f)) * scale;
}

inline float smoeq4_dequant(
    device const uint8_t* packed,
    device const half*    scales,
    uint wi,
    uint group_size)
{
    uint    pack_idx  = wi >> 2;                        // wi / 4
    uint    bit_shift = (wi & 3u) << 1u;               // (wi % 4) * 2
    uint    group_idx = wi / group_size;

    uint8_t code  = (packed[pack_idx] >> bit_shift) & 0x3u;
    float   scale = float(scales[group_idx]);

    // Affine mapping: centre 0-3 around 1.5, then scale
    return ((float(code) - 1.5f) * 0.666666667f) * scale;
}

// ── Fused Gate+Up+Down FFN kernel ────────────────────────────
//
// Buffer bindings (must match metal_bridge.mm exactly):
//   [0] gate_packed   — 2-bit packed gate projection weights
//   [1] gate_scales   — fp16 scale factors for gate
//   [2] up_packed     — 2-bit packed up projection weights
//   [3] up_scales     — fp16 scale factors for up
//   [4] down_packed   — 2-bit packed down projection weights
//   [5] down_scales   — fp16 scale factors for down
//   [6] input         — float32 input activation vector (cols elements)
//   [7] hidden        — float32 scratch: intermediate after gate+up+SiLU
//                       (rows elements; caller pre-zeros if needed)
//   [8] output        — float32 output vector (rows elements for down)
//   [9] params        — FusedFFNParams
//
// Pass 1 (gate + up → hidden) and Pass 2 (down → output) are
// separate dispatches so the hidden buffer is fully committed
// before any down-projection thread reads it. The bridge schedules
// them as two back-to-back compute passes in the same command buffer.
//
kernel void smoe_gate_up(
    device const uint8_t*      gate_packed  [[buffer(0)]],
    device const half*         gate_scales  [[buffer(1)]],
    device const uint8_t*      up_packed    [[buffer(2)]],
    device const half*         up_scales    [[buffer(3)]],
    device const float*        input        [[buffer(6)]],
    device       float*        hidden       [[buffer(7)]],
    constant FusedFFNParams&   params       [[buffer(9)]],
    uint                       gid          [[thread_position_in_grid]],
    uint                       tid          [[thread_index_in_threadgroup]],
    threadgroup float*         tg_input     [[threadgroup(0)]])
{
    // ── Phase A: cooperatively load the input vector into SRAM ──
    // Each threadgroup covers a TGROUP_SIZE-wide tile of `input`.
    // We iterate over input tiles until the whole vector is cached.
    //
    // NOTE: For cols > TGROUP_SIZE we reload tiles. Because SMOE-Q2
    // DeepSeek experts are narrow (cols typically 2048–7168), the
    // entire input vector may fit in a single tile on most configs.
    uint row = gid;                           // output element index
    if (row >= params.rows) return;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    uint tiles = (params.cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        uint col      = col_base + tid;

        // Load this thread's element into shared memory
        tg_input[tid] = (col < params.cols) ? input[col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate dot-products for all cols in this tile
        uint tile_end = min(TGROUP_SIZE, params.cols - col_base);

        // Unroll x4 over the natural 4-codes-per-byte boundary
        uint wi_base = row * params.cols + col_base;

        uint k = 0;
        if (params.bits == 4) {
            for (; k + 1 < tile_end; k += 2) {
                uint wi = wi_base + k;
                uint pidx = wi >> 1;
                uint8_t gbyte = gate_packed[pidx];
                uint8_t ubyte = up_packed[pidx];

                float gs = float(gate_scales[wi / params.group_size]);
                float us = float(up_scales[wi  / params.group_size]);

                for (uint b = 0; b < 2; ++b) {
                    float gcode = float((gbyte >> (b * 4)) & 0xFu);
                    float ucode = float((ubyte >> (b * 4)) & 0xFu);
                    float gw    = (gcode - 7.5f) * (1.0f / 7.5f) * gs;
                    float uw    = (ucode - 7.5f) * (1.0f / 7.5f) * us;
                    float x_k   = tg_input[k + b];
                    gate_acc   += gw * x_k;
                    up_acc     += uw * x_k;
                }
            }
            for (; k < tile_end; ++k) {
                uint wi = wi_base + k;
                gate_acc += smoeq4_dequant(gate_packed, gate_scales, wi, params.group_size) * tg_input[k];
                up_acc   += smoeq4_dequant(up_packed,   up_scales,   wi, params.group_size) * tg_input[k];
            }
        } else {
            for (; k + 3 < tile_end; k += 4) {
                uint wi = wi_base + k;
                uint pidx = wi >> 2;
                uint8_t gbyte = gate_packed[pidx];
                uint8_t ubyte = up_packed[pidx];

                float gs = float(gate_scales[wi / params.group_size]);
                float us = float(up_scales[wi  / params.group_size]);

                for (uint b = 0; b < 4; ++b) {
                    float gcode = float((gbyte >> (b * 2)) & 0x3u);
                    float ucode = float((ubyte >> (b * 2)) & 0x3u);
                    float gw    = (gcode - 1.5f) * (1.0f / 1.5f) * gs;
                    float uw    = (ucode - 1.5f) * (1.0f / 1.5f) * us;
                    float x_k   = tg_input[k + b];
                    gate_acc   += gw * x_k;
                    up_acc     += uw * x_k;
                }
            }
            for (; k < tile_end; ++k) {
                uint wi = wi_base + k;
                gate_acc += smoeq2_dequant(gate_packed, gate_scales, wi, params.group_size) * tg_input[k];
                up_acc   += smoeq2_dequant(up_packed,   up_scales,   wi, params.group_size) * tg_input[k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase B: fused SiLU gate ─────────────────────────────
    hidden[row] = silu(gate_acc) * up_acc;
}

// ── Down projection: hidden[rows] → output[rows] ─────────────
//
// One thread computes one element of the output vector by taking
// the dot-product of the corresponding down-projection row with
// the hidden activation vector.
//
kernel void smoe_down(
    device const uint8_t*      down_packed  [[buffer(4)]],
    device const half*         down_scales  [[buffer(5)]],
    device const float*        hidden       [[buffer(7)]],
    device       float*        output       [[buffer(8)]],
    constant FusedFFNParams&   params       [[buffer(9)]],
    uint                       gid          [[thread_position_in_grid]],
    uint                       tid          [[thread_index_in_threadgroup]],
    threadgroup float*         tg_hidden    [[threadgroup(0)]])
{
    uint row = gid;
    if (row >= params.cols) return;

    float acc   = 0.0f;
    uint  tiles = (params.rows + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        uint col      = col_base + tid;

        tg_hidden[tid] = (col < params.rows) ? hidden[col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_end = min(TGROUP_SIZE, params.rows - col_base);
        uint wi_base  = row * params.rows + col_base;

        uint k = 0;
        if (params.bits == 4) {
            for (; k + 1 < tile_end; k += 2) {
                uint    pidx  = (wi_base + k) >> 1;
                uint8_t dbyte = down_packed[pidx];
                float   ds    = float(down_scales[(wi_base + k) / params.group_size]);

                for (uint b = 0; b < 2; ++b) {
                    float dcode = float((dbyte >> (b * 4)) & 0xFu);
                    float dw    = (dcode - 7.5f) * (1.0f / 7.5f) * ds;
                    acc        += dw * tg_hidden[k + b];
                }
            }
            for (; k < tile_end; ++k) {
                uint wi = wi_base + k;
                acc += smoeq4_dequant(down_packed, down_scales, wi, params.group_size) * tg_hidden[k];
            }
        } else {
            for (; k + 3 < tile_end; k += 4) {
                uint    pidx  = (wi_base + k) >> 2;
                uint8_t dbyte = down_packed[pidx];
                float   ds    = float(down_scales[(wi_base + k) / params.group_size]);

                for (uint b = 0; b < 4; ++b) {
                    float dcode = float((dbyte >> (b * 2)) & 0x3u);
                    float dw    = (dcode - 1.5f) * (1.0f / 1.5f) * ds;
                    acc        += dw * tg_hidden[k + b];
                }
            }
            for (; k < tile_end; ++k) {
                uint wi = wi_base + k;
                acc += smoeq2_dequant(down_packed, down_scales, wi, params.group_size) * tg_hidden[k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[row] = acc;
}

// ── Scout Matrix-Vector Multiplication: weight[rows x cols] x input_vec[cols] → output_vec[rows] ──
kernel void scout_matvec(
    device const float*  weight       [[buffer(0)]],
    device const float*  input_vec    [[buffer(1)]],
    device       float*  output_vec   [[buffer(2)]],
    constant     uint2&  dims         [[buffer(3)]], // x = rows, y = cols
    uint                 row          [[thread_position_in_grid]],
    uint                 tid          [[thread_index_in_threadgroup]],
    threadgroup  float*  tg_input     [[threadgroup(0)]])
{
    uint rows = dims.x;
    uint cols = dims.y;

    // Load input_vec into threadgroup memory
    for (uint i = tid; i < cols; i += TGROUP_SIZE) {
        tg_input[i] = input_vec[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= rows) return;

    // Compute dot product
    float acc = 0.0f;
    device const float* row_ptr = weight + row * cols;
    uint c = 0;
    for (; c + 3 < cols; c += 4) {
        acc += row_ptr[c]     * tg_input[c];
        acc += row_ptr[c + 1] * tg_input[c + 1];
        acc += row_ptr[c + 2] * tg_input[c + 2];
        acc += row_ptr[c + 3] * tg_input[c + 3];
    }
    for (; c < cols; ++c) {
        acc += row_ptr[c] * tg_input[c];
    }
    output_vec[row] = acc;
}

