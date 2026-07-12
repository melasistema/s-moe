// ═══════════════════════════════════════════════════════════════
// metal_bridge.mm — S-MoE Engine · Objective-C++ Metal Bridge
// ═══════════════════════════════════════════════════════════════
// Phase 3 — Week 3
//
// Architecture:
//   JIT compile the MSL source via newLibraryWithSource:
//   Build two PSOs: smoe_gate_up and smoe_down.
//   Allocate two ping-pong MTLBuffers in Shared (UMA) storage.
//   Dispatch: encode gate_up pass → barrier → encode down pass
//             → commit → waitUntilCompleted.
//   Swap: plain id<MTLBuffer> pointer exchange, atomic flag update.
//
// Design invariants (matches GEMINI.md):
//   ① No malloc/new in the hot dispatch path — all resources
//     pre-allocated in smoe_metal_init().
//   ② UMA buffers in MTLResourceStorageModeShared: pread() from
//     the I/O streamer and the GPU shader access the same pages.
//   ③ Synchronisation via std::atomic, never OS mutexes.
// ═══════════════════════════════════════════════════════════════

#import  <Foundation/Foundation.h>
#import  <Metal/Metal.h>

#include "metal_bridge.h"
#include "../common.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cstdarg>
#include <thread>

static void info_printf(const char* format, ...) {
    const char* sm_dbg = std::getenv("SMOE_DEBUG");
    if (sm_dbg && (std::strcmp(sm_dbg, "1") == 0 || std::strcmp(sm_dbg, "true") == 0)) {
        va_list args;
        va_start(args, format);
        std::vfprintf(stderr, format, args);
        va_end(args);
    }
}

// ── Embedded MSL source ───────────────────────────────────────
// Read at compile time from the adjacent .metal file.
// We embed it as a raw string literal so the binary is fully
// self-contained (Option B decision: JIT on first launch).
// The string is derived from kernels.metal verbatim — any change
// to that file must be reflected here.
static const char* kMetalSource = R"MSL(
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

struct Dims {
    uint32_t rows;
    uint32_t cols;
};

inline float bf16_to_f32(ushort bf) {
    uint val = uint(bf) << 16;
    return as_type<float>(val);
}

// ── Tuning constants ─────────────────────────────────────────
// 256 threads per threadgroup: on Apple M-series GPU a 256-thread
// occupancy point typically saturates the SIMD width (32) × 8 waves.
constant uint TGROUP_SIZE = 256;

// ── Kernel argument buffer layout ────────────────────────────
struct FusedFFNParams {
    uint gate_rows;   // gate/up weight matrix rows  (= ffn_dim)
    uint gate_cols;   // gate/up weight matrix cols  (= d_model)
    uint down_rows;   // down weight matrix rows     (= d_model)
    uint down_cols;   // down weight matrix cols     (= ffn_dim)
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

    // Affine mapping: centre 0–3 around 1.5, then scale
    return (float(code) - 1.5f) * (1.0f / 1.5f) * scale;
}

inline float smoeq4_dequant(
    device const uint8_t* packed,
    device const half*    scales,
    uint wi,
    uint group_size)
{
    uint    pack_idx  = wi >> 1;                        // wi / 2
    uint    bit_shift = (wi & 1u) << 2u;               // (wi % 2) * 4
    uint    group_idx = wi / group_size;

    uint8_t code  = (packed[pack_idx] >> bit_shift) & 0xFu;
    float   scale = float(scales[group_idx]);

    // Affine mapping: centre 0-15 around 7.5, then scale
    return (float(code) - 7.5f) * (1.0f / 7.5f) * scale;
}

// ── Fused Gate+Up+Down FFN kernels (simdgroup-per-row) ───────
//
// Buffer bindings (must match metal_bridge.mm exactly):
//   [0] gate_packed   — packed gate projection weights
//   [1] gate_scales   — fp16 scale factors for gate
//   [2] up_packed     — packed up projection weights
//   [3] up_scales     — fp16 scale factors for up
//   [4] down_packed   — packed down projection weights
//   [5] down_scales   — fp16 scale factors for down
//   [6] input         — float32 input activation vector (cols elements)
//   [7] hidden        — float32 scratch: intermediate after gate+up+SiLU
//   [8] output        — float32 output vector (rows elements for down)
//   [9] params        — FusedFFNParams
//
// Pass 1 (gate + up → hidden) and Pass 2 (down → output) are
// separate dispatches so the hidden buffer is fully committed
// before any down-projection thread reads it. The bridge schedules
// them as two back-to-back compute passes in the same command buffer.
//
// DISPATCH CONTRACT (changed 2026-07-12):
//   grid            : ceil(rows / ROWS_PER_TG) threadgroups
//   threadsPerTG    : TGROUP_SIZE (256 = 8 simdgroups × 32 lanes)
//   threadgroup mem : cols × sizeof(float)  — the WHOLE staged input
//                     vector (16 KB at d_model 4096 — under the 32 KB
//                     limit up to cols 8192)
//
// One SIMDGROUP owns one output row; its 32 lanes read consecutive
// uint4 chunks of the packed row — 512 coalesced bytes per step. The
// old thread-per-row layout made the 32 lanes of a simdgroup read
// addresses ~2 KB apart one byte at a time: measured 22.6 GB/s
// effective vs 158 GB/s for this layout on M4 Pro
// (scratch/kernel_bench_results.md, 7.0× per expert group).
//
// Dequantisation is folded into a per-chunk affine epilogue: with
//   w = (code − Z)·(1/Z)·s        (Z = 7.5 at Q4, 1.5 at Q2)
//   Σ w·x = s·( (Σ code·x)/Z − Σ x )
// so the inner loop is one fma(code4, x4, acc) per 4 codes and the
// scale is read once per 16-byte chunk instead of once per 2 codes.
//
// Layout assumptions (hold for every .smoe vault — dims are multiples
// of 256, group_size 64):
//   • cols % 32 == 0 (Q4) / % 64 == 0 (Q2) — no chunk tail handling
//   • group_size % 32 == 0 (Q4) / % 64 == 0 (Q2) — a 16-byte chunk
//     never straddles a scale group
//   • packed rows are 16-byte aligned (row stride = cols/2 B at Q4,
//     cols/4 B at Q2 — multiples of 16 when cols % 64 == 0)

constant uint ROWS_PER_TG = 8;      // TGROUP_SIZE / 32 lanes

// ── One quantized row · staged input vector, simdgroup-cooperative ──
// Each lane owns the uint4 chunks c = lane, lane+32, … of the row.
// Returns the full row dot on EVERY lane (simd_sum broadcast).
inline float qrow_dot(device const uint8_t*    packed,
                      device const half*       scales,
                      threadgroup const float* x,
                      uint row, uint cols,
                      uint group_size, uint bits, uint lane)
{
    float acc = 0.0f;

    if (bits == 4) {
        // 16-byte chunk = 32 codes; row stride = cols/2 bytes.
        device const uint4* rp =
            (device const uint4*)(packed + (ulong)row * (cols >> 1));
        device const half* srow = scales + (ulong)row * (cols / group_size);
        const uint nchunks = cols >> 5;

        for (uint c = lane; c < nchunks; c += 32) {
            uint4 v = rp[c];
            float s = float(srow[(c << 5) / group_size]);
            threadgroup const float4* xv =
                (threadgroup const float4*)(x + (c << 5));

            float4 a  = float4(0.0f);
            float4 xs = float4(0.0f);
            for (uint k = 0; k < 4; ++k) {
                uint u = v[k];                       // 8 codes
                float4 clo = float4(float( u        & 0xFu),
                                    float((u >>  4) & 0xFu),
                                    float((u >>  8) & 0xFu),
                                    float((u >> 12) & 0xFu));
                float4 chi = float4(float((u >> 16) & 0xFu),
                                    float((u >> 20) & 0xFu),
                                    float((u >> 24) & 0xFu),
                                    float((u >> 28) & 0xFu));
                float4 xlo = xv[k * 2];
                float4 xhi = xv[k * 2 + 1];
                a   = fma(clo, xlo, a);
                a   = fma(chi, xhi, a);
                xs += xlo + xhi;
            }
            float A = a.x + a.y + a.z + a.w;
            float X = xs.x + xs.y + xs.z + xs.w;
            acc = fma(s, A * (1.0f / 7.5f) - X, acc);
        }
    } else {
        // Q2: 16-byte chunk = 64 codes; row stride = cols/4 bytes.
        device const uint4* rp =
            (device const uint4*)(packed + (ulong)row * (cols >> 2));
        device const half* srow = scales + (ulong)row * (cols / group_size);
        const uint nchunks = cols >> 6;

        for (uint c = lane; c < nchunks; c += 32) {
            uint4 v = rp[c];
            float s = float(srow[(c << 6) / group_size]);
            threadgroup const float4* xv =
                (threadgroup const float4*)(x + (c << 6));

            float4 a  = float4(0.0f);
            float4 xs = float4(0.0f);
            for (uint k = 0; k < 4; ++k) {
                uint u = v[k];                       // 16 codes
                for (uint g = 0; g < 4; ++g) {       // 4 codes per step
                    float4 cq = float4(float((u >> (8*g    )) & 3u),
                                       float((u >> (8*g + 2)) & 3u),
                                       float((u >> (8*g + 4)) & 3u),
                                       float((u >> (8*g + 6)) & 3u));
                    float4 xq = xv[k * 4 + g];
                    a   = fma(cq, xq, a);
                    xs += xq;
                }
            }
            float A = a.x + a.y + a.z + a.w;
            float X = xs.x + xs.y + xs.z + xs.w;
            acc = fma(s, A * (1.0f / 1.5f) - X, acc);
        }
    }
    return simd_sum(acc);
}

// ── Two quantized rows (gate + up) · staged input, one pass ─────
// Both matrices share x, so Σx and the xv loads are paid once for
// the two dots (measured ~17% over two qrow_dot passes).
inline void qrow_dot2(device const uint8_t*    g_packed,
                      device const half*       g_scales,
                      device const uint8_t*    u_packed,
                      device const half*       u_scales,
                      threadgroup const float* x,
                      uint row, uint cols,
                      uint group_size, uint bits, uint lane,
                      thread float& g_out, thread float& u_out)
{
    float gacc = 0.0f, uacc = 0.0f;

    if (bits == 4) {
        device const uint4* gp =
            (device const uint4*)(g_packed + (ulong)row * (cols >> 1));
        device const uint4* up =
            (device const uint4*)(u_packed + (ulong)row * (cols >> 1));
        device const half* gsrow = g_scales + (ulong)row * (cols / group_size);
        device const half* usrow = u_scales + (ulong)row * (cols / group_size);
        const uint nchunks = cols >> 5;

        for (uint c = lane; c < nchunks; c += 32) {
            uint4 vg = gp[c];
            uint4 vu = up[c];
            float sg = float(gsrow[(c << 5) / group_size]);
            float su = float(usrow[(c << 5) / group_size]);
            threadgroup const float4* xv =
                (threadgroup const float4*)(x + (c << 5));

            float4 ag = float4(0.0f), au = float4(0.0f), xs = float4(0.0f);
            for (uint k = 0; k < 4; ++k) {
                uint ug = vg[k];
                uint uu = vu[k];
                float4 xlo = xv[k * 2];
                float4 xhi = xv[k * 2 + 1];
                ag = fma(float4(float( ug        & 0xFu),
                                float((ug >>  4) & 0xFu),
                                float((ug >>  8) & 0xFu),
                                float((ug >> 12) & 0xFu)), xlo, ag);
                ag = fma(float4(float((ug >> 16) & 0xFu),
                                float((ug >> 20) & 0xFu),
                                float((ug >> 24) & 0xFu),
                                float((ug >> 28) & 0xFu)), xhi, ag);
                au = fma(float4(float( uu        & 0xFu),
                                float((uu >>  4) & 0xFu),
                                float((uu >>  8) & 0xFu),
                                float((uu >> 12) & 0xFu)), xlo, au);
                au = fma(float4(float((uu >> 16) & 0xFu),
                                float((uu >> 20) & 0xFu),
                                float((uu >> 24) & 0xFu),
                                float((uu >> 28) & 0xFu)), xhi, au);
                xs += xlo + xhi;
            }
            float X = xs.x + xs.y + xs.z + xs.w;
            gacc = fma(sg, (ag.x + ag.y + ag.z + ag.w) * (1.0f / 7.5f) - X, gacc);
            uacc = fma(su, (au.x + au.y + au.z + au.w) * (1.0f / 7.5f) - X, uacc);
        }
        g_out = simd_sum(gacc);
        u_out = simd_sum(uacc);
        return;
    }

    // Q2: reuse the single-row path twice (no Q2 vault exists yet to
    // measure against; fuse here too if Q2 becomes a hot path).
    g_out = qrow_dot(g_packed, g_scales, x, row, cols, group_size, bits, lane);
    u_out = qrow_dot(u_packed, u_scales, x, row, cols, group_size, bits, lane);
}

// ── Fused gate+up+SiLU: hidden[row] = silu(gate·x) * (up·x) ─────
kernel void smoe_gate_up(
    device const uint8_t*      gate_packed  [[buffer(0)]],
    device const half*         gate_scales  [[buffer(1)]],
    device const uint8_t*      up_packed    [[buffer(2)]],
    device const half*         up_scales    [[buffer(3)]],
    device const float*        input        [[buffer(6)]],
    device       float*        hidden       [[buffer(7)]],
    constant FusedFFNParams&   params       [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* xshared [[threadgroup(0)]])
{
    // Stage the input vector once per threadgroup (shared by 8 rows).
    for (uint i = tid; i < params.gate_cols; i += TGROUP_SIZE) {
        xshared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tgid * ROWS_PER_TG + sgi;
    if (row >= params.gate_rows) return;   // whole simdgroup exits together

    float g, u;
    qrow_dot2(gate_packed, gate_scales, up_packed, up_scales, xshared,
              row, params.gate_cols, params.group_size, params.bits, lane, g, u);
    if (lane == 0) {
        hidden[row] = silu(g) * u;
    }
}

// ── Down projection: output[row] = down_row · hidden ────────────
kernel void smoe_down(
    device const uint8_t*      down_packed  [[buffer(4)]],
    device const half*         down_scales  [[buffer(5)]],
    device const float*        hidden       [[buffer(7)]],
    device       float*        output       [[buffer(8)]],
    constant FusedFFNParams&   params       [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* xshared [[threadgroup(0)]])
{
    for (uint i = tid; i < params.down_cols; i += TGROUP_SIZE) {
        xshared[i] = hidden[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tgid * ROWS_PER_TG + sgi;
    if (row >= params.down_rows) return;

    float acc = qrow_dot(down_packed, down_scales, xshared,
                         row, params.down_cols, params.group_size, params.bits, lane);
    if (lane == 0) {
        output[row] = acc;
    }
}

// ── Scout Matrix-Vector Multiplication: weight[rows x cols] x input_vec[cols] → output_vec[rows] ──
kernel void scout_matvec(
    device const float*  weight       [[buffer(0)]],
    device const float*  input_vec    [[buffer(1)]],
    device       float*  output_vec   [[buffer(2)]],
    constant     uint2&  dims         [[buffer(3)]], // x = rows, y = cols
    uint                 row          [[thread_position_in_grid]],
    uint                 tid          [[thread_index_in_threadgroup]],
    uint                 threads_per_tg [[threads_per_threadgroup]],
    threadgroup  float*  tg_input     [[threadgroup(0)]])
{
    uint rows = dims.x;
    uint cols = dims.y;

    // Load input_vec into threadgroup memory
    for (uint i = tid; i < cols; i += threads_per_tg) {
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


// ── Scout BF16 Matvec ─────────────────────────────────────────
kernel void scout_matvec_bf16(
    device const uint16_t* weights [[buffer(0)]],
    device const float*    input   [[buffer(1)]],
    device       float*    output  [[buffer(2)]],
    constant     Dims&     dims    [[buffer(3)]],
    uint                   gid     [[thread_position_in_grid]],
    uint                   tid     [[thread_index_in_threadgroup]],
    uint                   threads_per_tg [[threads_per_threadgroup]],
    threadgroup  float*    tg_input [[threadgroup(0)]])
{
    uint row = gid;
    float acc = 0.0f;
    uint tiles = (dims.cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        for (uint i = tid; i < TGROUP_SIZE; i += threads_per_tg) {
            uint col = col_base + i;
            tg_input[i] = (col < dims.cols) ? input[col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < dims.rows) {
            uint tile_end = min(TGROUP_SIZE, dims.cols - col_base);
            for (uint k = 0; k < tile_end; ++k) {
                uint col = col_base + k;
                ushort bf = weights[row * dims.cols + col];
                float w = bf16_to_f32(bf);
                acc += w * tg_input[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < dims.rows) {
        output[row] = acc;
    }
}

// ═══════════════════════════════════════════════════════════════
// Decode attention (single token) — one command buffer per layer
// ═══════════════════════════════════════════════════════════════
// Two kernels bridge the gap between the QKV and O projections so
// the whole attention block runs GPU-resident with ONE CPU sync per
// layer instead of two (the CPU previously did QK-norm/RoPE/attention
// between two synchronous matvec roundtrips).

struct AttnPrepParams {
    uint head_dim;      // per-head dim (Qwen3: 128)
    uint num_heads;     // query heads
    uint num_kv_heads;  // KV heads (GQA)
    uint rope_half;     // head_dim / 2
    uint slot;          // KV ring write slot (ctx_pos % attn_ctx)
    uint kv_dim;        // num_kv_heads * head_dim
    uint has_qk_norm;   // 0 → skip per-head RMS norm entirely
};

// Per-head QK RMS-norm + RoPE + KV-cache append.
// Grid: (num_heads + num_kv_heads) threadgroups × head_dim threads.
// Threadgroups < num_heads process a Q head in place; the rest
// process a K head and write the rotated K row plus the raw V row
// straight into the layer's KV ring at `slot`.
// Threadgroup memory: (head_dim + 32) floats — staged head vector
// followed by per-simdgroup reduction scratch.
kernel void attn_prep(
    device       float*    qbuf      [[buffer(0)]],
    device       float*    kbuf      [[buffer(1)]],
    device const float*    vbuf      [[buffer(2)]],
    device const uint16_t* q_norm_w  [[buffer(3)]],
    device const uint16_t* k_norm_w  [[buffer(4)]],
    device const float*    rope_cos  [[buffer(5)]],
    device const float*    rope_sin  [[buffer(6)]],
    device       float*    k_cache   [[buffer(7)]],
    device       float*    v_cache   [[buffer(8)]],
    constant AttnPrepParams& p       [[buffer(9)]],
    uint hg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint ntid [[threads_per_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    const bool is_q = (hg < p.num_heads);
    const uint h    = is_q ? hg : (hg - p.num_heads);
    device float* head = is_q ? (qbuf + h * p.head_dim)
                              : (kbuf + h * p.head_dim);

    threadgroup float* vec = tg;
    threadgroup float* red = tg + p.head_dim;

    float x = (tid < p.head_dim) ? head[tid] : 0.0f;

    if (p.has_qk_norm != 0) {
        // Must match smoe::rms_norm_bf16: mean-of-squares + 1e-6.
        // simd_sum reassociates the reduction — not bit-exact vs CPU.
        float ss = simd_sum(x * x);
        if (lane == 0) red[sgi] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tot = 0.0f;
            uint nsg = (ntid + 31u) / 32u;
            for (uint i = 0; i < nsg; ++i) tot += red[i];
            red[0] = 1.0f / sqrt(tot / float(p.head_dim) + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = red[0];
        device const uint16_t* nw = is_q ? q_norm_w : k_norm_w;
        if (tid < p.head_dim) x = (x * inv) * bf16_to_f32(nw[tid]);
    }
    if (tid < p.head_dim) vec[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RoPE pair rotation: (d, d + rope_half) share angle d.
    float rot = x;
    if (tid < p.rope_half) {
        rot = vec[tid] * rope_cos[tid] - vec[tid + p.rope_half] * rope_sin[tid];
    } else if (tid < p.head_dim) {
        uint d = tid - p.rope_half;
        rot = vec[d] * rope_sin[d] + vec[tid] * rope_cos[d];
    }

    if (tid < p.head_dim) {
        if (is_q) {
            head[tid] = rot;
        } else {
            uint co = p.slot * p.kv_dim + h * p.head_dim + tid;
            k_cache[co] = rot;
            v_cache[co] = vbuf[h * p.head_dim + tid];
        }
    }
}

struct AttnDecodeParams {
    uint  head_dim;
    uint  num_heads;
    uint  heads_per_kv;  // num_heads / num_kv_heads (GQA fan-in)
    uint  kv_dim;
    uint  slot;          // ring slot of the CURRENT token
    uint  valid;         // attended positions incl. current (≤ attn_ctx)
    uint  attn_ctx;      // KV ring capacity
    float scale;         // 1/sqrt(head_dim)
};

// Single-token GQA attention over the KV ring.
// Grid: num_heads threadgroups × TGROUP_SIZE threads.
// Threadgroup memory: (head_dim + valid + 32) floats —
//   staged Q head | score row | per-simdgroup reduction scratch.
// valid ≤ 4096 keeps this under the 32 KB threadgroup limit.
kernel void attn_decode(
    device const float* qbuf     [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    device       float* attn_out [[buffer(3)]],
    constant AttnDecodeParams& p [[buffer(4)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint ntid [[threads_per_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* tg [[threadgroup(0)]])
{
    threadgroup float* qvec   = tg;
    threadgroup float* scores = tg + p.head_dim;
    threadgroup float* red    = scores + p.valid;

    const uint kv_off = (h / p.heads_per_kv) * p.head_dim;

    for (uint d = tid; d < p.head_dim; d += ntid) {
        qvec[d] = qbuf[h * p.head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: position i counts back from the current slot.
    float lmax = -INFINITY;
    for (uint i = tid; i < p.valid; i += ntid) {
        uint ki = (p.slot + p.attn_ctx - i) % p.attn_ctx;
        device const float* krow = k_cache + ki * p.kv_dim + kv_off;
        float acc = 0.0f;
        for (uint d = 0; d < p.head_dim; ++d) acc += qvec[d] * krow[d];
        acc *= p.scale;
        scores[i] = acc;
        lmax = max(lmax, acc);
    }
    lmax = simd_max(lmax);
    if (lane == 0) red[sgi] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = red[0];
        uint nsg = (ntid + 31u) / 32u;
        for (uint i = 1; i < nsg; ++i) m = max(m, red[i]);
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup); // red[0] reused below

    float lsum = 0.0f;
    for (uint i = tid; i < p.valid; i += ntid) {
        float e = exp(scores[i] - gmax);
        scores[i] = e;
        lsum += e;
    }
    lsum = simd_sum(lsum);
    if (lane == 0) red[sgi] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        uint nsg = (ntid + 31u) / 32u;
        for (uint i = 0; i < nsg; ++i) s += red[i];
        red[0] = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = red[0];

    // Weighted V sum: thread d owns output dim d; consecutive threads
    // read consecutive addresses of each V row (coalesced).
    for (uint d = tid; d < p.head_dim; d += ntid) {
        float acc = 0.0f;
        for (uint i = 0; i < p.valid; ++i) {
            uint vi = (p.slot + p.attn_ctx - i) % p.attn_ctx;
            acc += scores[i] * v_cache[vi * p.kv_dim + kv_off + d];
        }
        attn_out[h * p.head_dim + d] = acc * inv_sum;
    }
}

)MSL";

// Token-batch fused FFN kernels, concatenated after kMetalSource at JIT
// time so they can reuse TGROUP_SIZE / silu / smoeq*_dequant.
// Mirror of the batch section appended to kernels.metal — keep in sync.
static const char* kMetalBatchSource = R"MSL(

struct FusedFFNBatchParams {
    uint gate_rows;
    uint gate_cols;
    uint down_rows;
    uint down_cols;
    uint group_size;
    uint bits;
    uint batch;
};

kernel void smoe_gate_up_batch(
    device const uint8_t*          gate_packed [[buffer(0)]],
    device const half*             gate_scales [[buffer(1)]],
    device const uint8_t*          up_packed   [[buffer(2)]],
    device const half*             up_scales   [[buffer(3)]],
    device const float*            input_mat   [[buffer(6)]],
    device       float*            hidden_mat  [[buffer(7)]],
    constant FusedFFNBatchParams&  params      [[buffer(9)]],
    uint2                          tg_pos      [[threadgroup_position_in_grid]],
    uint                           tid         [[thread_index_in_threadgroup]],
    uint2                          tg_dim      [[threads_per_threadgroup]],
    threadgroup float*             tg_input    [[threadgroup(0)]])
{
    const uint b   = tg_pos.y;
    const uint row = tg_pos.x * tg_dim.x + tid;
    device const float* input = input_mat + b * params.gate_cols;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    uint tiles = (params.gate_cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        for (uint i = tid; i < TGROUP_SIZE; i += tg_dim.x) {
            uint col = col_base + i;
            tg_input[i] = (col < params.gate_cols) ? input[col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_end = min(TGROUP_SIZE, params.gate_cols - col_base);
        if (row < params.gate_rows) {
            uint wi_base = row * params.gate_cols + col_base;
            for (uint k = 0; k < tile_end; ++k) {
                uint wi = wi_base + k;
                float gw, uw;
                if (params.bits == 4) {
                    gw = smoeq4_dequant(gate_packed, gate_scales, wi, params.group_size);
                    uw = smoeq4_dequant(up_packed,   up_scales,   wi, params.group_size);
                } else {
                    gw = smoeq2_dequant(gate_packed, gate_scales, wi, params.group_size);
                    uw = smoeq2_dequant(up_packed,   up_scales,   wi, params.group_size);
                }
                gate_acc += gw * tg_input[k];
                up_acc   += uw * tg_input[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.gate_rows) {
        hidden_mat[b * params.gate_rows + row] = silu(gate_acc) * up_acc;
    }
}

kernel void smoe_down_batch(
    device const uint8_t*          down_packed [[buffer(4)]],
    device const half*             down_scales [[buffer(5)]],
    device const float*            hidden_mat  [[buffer(7)]],
    device       float*            output_mat  [[buffer(8)]],
    constant FusedFFNBatchParams&  params      [[buffer(9)]],
    uint2                          tg_pos      [[threadgroup_position_in_grid]],
    uint                           tid         [[thread_index_in_threadgroup]],
    uint2                          tg_dim      [[threads_per_threadgroup]],
    threadgroup float*             tg_hidden   [[threadgroup(0)]])
{
    const uint b   = tg_pos.y;
    const uint row = tg_pos.x * tg_dim.x + tid;
    device const float* hidden = hidden_mat + b * params.down_cols;

    float acc  = 0.0f;
    uint tiles = (params.down_cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        for (uint i = tid; i < TGROUP_SIZE; i += tg_dim.x) {
            uint col = col_base + i;
            tg_hidden[i] = (col < params.down_cols) ? hidden[col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_end = min(TGROUP_SIZE, params.down_cols - col_base);
        if (row < params.down_rows) {
            uint wi_base = row * params.down_cols + col_base;
            for (uint k = 0; k < tile_end; ++k) {
                uint wi = wi_base + k;
                float dw = (params.bits == 4)
                    ? smoeq4_dequant(down_packed, down_scales, wi, params.group_size)
                    : smoeq2_dequant(down_packed, down_scales, wi, params.group_size);
                acc += dw * tg_hidden[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.down_rows) {
        output_mat[b * params.down_rows + row] = acc;
    }
}

)MSL";

// ═══════════════════════════════════════════════════════════════
// SmoeMetalCtx — internal implementation struct
// ═══════════════════════════════════════════════════════════════
struct Dims {
    uint32_t rows;
    uint32_t cols;
};

struct SmoeMetalCtx {
    id<MTLDevice>               device       = nil;
    id<MTLCommandQueue>         queue        = nil;
    id<MTLLibrary>              library      = nil;
    id<MTLComputePipelineState> gate_up_pso  = nil;
    id<MTLComputePipelineState> down_pso     = nil;
    id<MTLComputePipelineState> scout_matvec_pso = nil;
    id<MTLComputePipelineState> scout_matvec_bf16_pso = nil;
    id<MTLComputePipelineState> attn_prep_pso   = nil;
    id<MTLComputePipelineState> attn_decode_pso = nil;
    id<MTLComputePipelineState> gate_up_batch_pso = nil;
    id<MTLComputePipelineState> down_batch_pso    = nil;

    // Token-batch fused FFN staging (layer-major prefill):
    //   batch_in  [SMOE_MAX_FFN_BATCH × gate_cols]
    //   batch_hid [SMOE_MAX_FFN_BATCH × gate_rows]
    //   batch_out [SMOE_MAX_FFN_BATCH × down_rows]
    // Dim cap: SMOE_MAX_FFN_BATCH_DIM floats per row.
    id<MTLBuffer>               batch_in_buf     = nil;
    id<MTLBuffer>               batch_hidden_buf = nil;
    id<MTLBuffer>               batch_out_buf    = nil;

    // Ping-pong UMA buffers
    // buf_a = active (GPU executing)
    // buf_b = passive (I/O streamer filling)
    id<MTLBuffer>               buf_a        = nil;  // active
    id<MTLBuffer>               buf_b        = nil;  // passive

    // Pre-allocated scratch buffer for the hidden activation vector.
    // Size = max_rows * sizeof(float). Allocated in smoe_metal_init.
    id<MTLBuffer>               hidden_buf   = nil;
    id<MTLBuffer>               output_buf   = nil;
    id<MTLBuffer>               input_buf    = nil;

    size_t                      slot_bytes   = 0;
    uint32_t                    max_rows     = 0;    // = slot_bytes / sizeof(float)

    // Atomic flag: true when buf_b is safe for the I/O streamer to fill.
    std::atomic<bool>           passive_ready { true };

    // Telemetry
    std::atomic<uint64_t>       dispatch_count { 0 };

    // Shared event for async GPU/CPU synchronisation
    id<MTLSharedEvent>          shared_event = nil;
    uint64_t                    signaled_value = 0;

    struct RegisteredBuffer {
        const void* ptr;
        size_t length;
        id<MTLBuffer> buffer;
    };
    std::vector<RegisteredBuffer> registered_buffers;
};

static id<MTLBuffer> get_registered_buffer(SmoeMetalCtx* ctx, const void* ptr, size_t sz, NSUInteger& out_offset) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& reg : ctx->registered_buffers) {
        uintptr_t base = reinterpret_cast<uintptr_t>(reg.ptr);
        if (p >= base && (p + sz) <= (base + reg.length)) {
            out_offset = p - base;
            return reg.buffer;
        }
    }
    
    uintptr_t aligned_p = p & ~(smoe::PAGE_SIZE - 1);
    out_offset = p - aligned_p;
    size_t aligned_sz = ((p + sz + (smoe::PAGE_SIZE - 1)) & ~(smoe::PAGE_SIZE - 1)) - aligned_p;
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:reinterpret_cast<void*>(aligned_p)
                                          length:aligned_sz
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
    if (buf) {
        ctx->registered_buffers.push_back({reinterpret_cast<const void*>(aligned_p), aligned_sz, buf});
    }
    return buf;
}

// ── Internal helpers ──────────────────────────────────────────

// Build a compute PSO from a function name in the already-compiled library.
static id<MTLComputePipelineState> make_pso(
    id<MTLDevice>  device,
    id<MTLLibrary> lib,
    const char*    fn_name,
    NSError**      err)
{
    NSString* name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: function '%s' not found in library.\n", fn_name);
        return nil;
    }
    return [device newComputePipelineStateWithFunction:fn error:err];
}

// ── Public API ────────────────────────────────────────────────

SmoeMetalCtx* smoe_metal_init(size_t slot_bytes) {
    // ── Step 1: acquire the default Metal device ──────────────
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        std::fprintf(stderr, "[smoe_metal] ERROR: No Metal device found.\n");
        return nullptr;
    }

    // ── Step 2: JIT-compile the embedded MSL source ───────────
    // This takes ~200–500 ms on first run; the compiled PSO is
    // cached by the Metal compiler for subsequent launches.
    info_printf("[smoe_metal] Compiling SMOE-Q2 kernels (JIT)...\n");

    NSError*  err    = nil;
    NSString* source = [[NSString stringWithUTF8String:kMetalSource]
        stringByAppendingString:[NSString stringWithUTF8String:kMetalBatchSource]];

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_1;
    opts.fastMathEnabled = YES;  // safe for our affine dequant formula

    id<MTLLibrary> lib = [dev newLibraryWithSource:source
                                           options:opts
                                             error:&err];
    if (!lib) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: MSL compilation failed:\n%s\n",
            [[err localizedDescription] UTF8String]);
        return nullptr;
    }
    info_printf("[smoe_metal] Kernel compilation OK.\n");

    // ── Step 3: build PSOs ────────────────────────────────────
    id<MTLComputePipelineState> gate_up_pso = make_pso(dev, lib, "smoe_gate_up", &err);
    if (!gate_up_pso) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: PSO 'smoe_gate_up' build failed:\n%s\n",
            [[err localizedDescription] UTF8String]);
        return nullptr;
    }

    id<MTLComputePipelineState> down_pso = make_pso(dev, lib, "smoe_down", &err);
    if (!down_pso) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: PSO 'smoe_down' build failed:\n%s\n",
            [[err localizedDescription] UTF8String]);
        return nullptr;
    }

    id<MTLComputePipelineState> scout_matvec_pso = make_pso(dev, lib, "scout_matvec", &err);
    if (!scout_matvec_pso) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: PSO 'scout_matvec' build failed:\n%s\n",
            [[err localizedDescription] UTF8String]);
        return nullptr;
    }

    // ── Step 4: allocate ping-pong UMA buffers ────────────────
    // MTLResourceStorageModeShared: CPU pread() and GPU shaders
    // access the same physical pages — zero-copy guaranteed.
    MTLResourceOptions uma = MTLResourceStorageModeShared;

    id<MTLBuffer> buf_a = [dev newBufferWithLength:slot_bytes options:uma];
    id<MTLBuffer> buf_b = [dev newBufferWithLength:slot_bytes options:uma];
    if (!buf_a || !buf_b) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: Failed to allocate ping-pong buffers "
            "(%zu bytes each).\n", slot_bytes);
        return nullptr;
    }

    // ── Step 5: allocate hidden-vector scratch buffer ─────────
    // Sized for the worst-case intermediate dim x 8 concurrent experts.
    uint32_t max_rows    = static_cast<uint32_t>(
        std::min(slot_bytes / sizeof(float), size_t(32768)));
    id<MTLBuffer> hidden = [dev newBufferWithLength:16 * max_rows * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_b = [dev newBufferWithLength:16 * max_rows * sizeof(float)
                                            options:uma];
    id<MTLBuffer> input_b = [dev newBufferWithLength:8192 * sizeof(float)
                                            options:uma];
    if (!hidden || !output_b || !input_b) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: Failed to allocate hidden/output/input scratch buffers.\n");
        return nullptr;
    }

    // ── Step 6: create command queue & shared event ──────────
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!queue) {
        std::fprintf(stderr, "[smoe_metal] ERROR: Failed to create command queue.\n");
        return nullptr;
    }
    id<MTLSharedEvent> shared_ev = [dev newSharedEvent];

    // ── Step 7: assemble context ──────────────────────────────
    SmoeMetalCtx* ctx = new SmoeMetalCtx();
    ctx->device      = dev;
    ctx->queue       = queue;
    ctx->library     = lib;
    ctx->gate_up_pso = gate_up_pso;
    ctx->down_pso    = down_pso;
    ctx->scout_matvec_pso = scout_matvec_pso;
    
    id<MTLComputePipelineState> scout_matvec_bf16_pso = make_pso(dev, lib, "scout_matvec_bf16", &err);
    if (!scout_matvec_bf16_pso) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: PSO 'scout_matvec_bf16' build failed:\n%s\n",
            [[err localizedDescription] UTF8String]);
        return nullptr;
    }
    ctx->scout_matvec_bf16_pso = scout_matvec_bf16_pso;

    ctx->attn_prep_pso   = make_pso(dev, lib, "attn_prep", &err);
    ctx->attn_decode_pso = make_pso(dev, lib, "attn_decode", &err);
    if (!ctx->attn_prep_pso || !ctx->attn_decode_pso) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: attention PSO build failed:\n%s\n",
            err ? [[err localizedDescription] UTF8String] : "(no detail)");
        return nullptr;
    }

    // Token-batch FFN: non-fatal if unavailable — callers fall back to
    // the per-token fused FFN path when these PSOs/buffers are nil.
    ctx->gate_up_batch_pso = make_pso(dev, lib, "smoe_gate_up_batch", &err);
    ctx->down_batch_pso    = make_pso(dev, lib, "smoe_down_batch", &err);
    if (ctx->gate_up_batch_pso && ctx->down_batch_pso) {
        // Row cap is SMOE_MAX_FFN_BATCH_DIM floats; ~4 MB per plane at
        // 64 × 16384 × 4 B. Shared-mode: CPU memcpy in, GPU writes out.
        size_t plane = size_t(SMOE_MAX_FFN_BATCH) * SMOE_MAX_FFN_BATCH_DIM * sizeof(float);
        ctx->batch_in_buf     = [dev newBufferWithLength:plane options:uma];
        ctx->batch_hidden_buf = [dev newBufferWithLength:plane options:uma];
        ctx->batch_out_buf    = [dev newBufferWithLength:plane options:uma];
        if (!ctx->batch_in_buf || !ctx->batch_hidden_buf || !ctx->batch_out_buf) {
            std::fprintf(stderr, "[smoe_metal] WARN: batch FFN staging alloc failed — per-token path only.\n");
            ctx->gate_up_batch_pso = nil;
            ctx->down_batch_pso    = nil;
        }
    } else {
        std::fprintf(stderr, "[smoe_metal] WARN: batch FFN PSOs unavailable — per-token path only.\n");
    }

    ctx->buf_a       = buf_a;
    ctx->buf_b       = buf_b;
    ctx->hidden_buf  = hidden;
    ctx->output_buf  = output_b;
    ctx->input_buf   = input_b;
    ctx->slot_bytes  = slot_bytes;
    ctx->max_rows    = max_rows;
    ctx->shared_event = shared_ev;
    ctx->signaled_value = 0;
    ctx->passive_ready.store(true, std::memory_order_release);
    ctx->dispatch_count.store(0, std::memory_order_relaxed);

    info_printf(
        "[smoe_metal] Init OK — slot=%.1f MB, hidden_scratch=%u floats.\n",
        double(slot_bytes) / (1024.0 * 1024.0), max_rows);

    return ctx;
}

void smoe_metal_destroy(SmoeMetalCtx* ctx) {
    if (!ctx) return;
    // ARC handles all id<MTL*> release; we only need to free the struct.
    delete ctx;
}

void* smoe_metal_passive_ptr(SmoeMetalCtx* ctx) {
    if (!ctx || !ctx->buf_b) return nullptr;
    return [ctx->buf_b contents];
}

size_t smoe_metal_slot_bytes(const SmoeMetalCtx* ctx) {
    return ctx ? ctx->slot_bytes : 0;
}

void* smoe_metal_fused_ffn(SmoeMetalCtx*   ctx,
                           const uint8_t*  packed_gate,
                           const uint16_t* scales_gate,
                           const uint8_t*  packed_up,
                           const uint16_t* scales_up,
                           const uint8_t*  packed_down,
                           const uint16_t* scales_down,
                           const float*    input_vec,
                           float*          hidden_vec,
                           float*          output_vec,
                           uint32_t        gate_rows,
                           uint32_t        gate_cols,
                           uint32_t        down_rows,
                           uint32_t        down_cols,
                           uint32_t        group_size,
                           uint32_t        bits,
                           uint32_t        expert_index)
{
    if (!ctx) return nullptr;

    // ── Build FusedFFNParams ──────────────────────────────────
    // Stack allocation — not in hot-path alloc-banned zone because
    // this struct is tiny and compiler-stack-allocated.
    struct FusedFFNParamsC { uint32_t gate_rows, gate_cols, down_rows, down_cols, group_size, bits; };
    FusedFFNParamsC params { gate_rows, gate_cols, down_rows, down_cols, group_size, bits };

    // ── Wrap raw pointers in MTLBuffers (no-copy) ─────────────
    @autoreleasepool {
        // Compute byte sizes for each sub-tensor (assuming gate/up are same shape and down is transpose)
        uint64_t packed_bytes = static_cast<uint64_t>(gate_rows) * gate_cols * bits / 8;
        uint64_t scale_bytes  = static_cast<uint64_t>(gate_rows) * gate_cols / group_size * sizeof(uint16_t);
        uint64_t input_bytes  = static_cast<uint64_t>(gate_cols)  * sizeof(float);
        uint64_t output_bytes = static_cast<uint64_t>(down_rows)  * sizeof(float);

        NSUInteger off_gp = 0, off_gs = 0, off_up = 0, off_us = 0, off_dp = 0, off_ds = 0;
        id<MTLBuffer> buf_gp = get_registered_buffer(ctx, packed_gate,  packed_bytes, off_gp);
        id<MTLBuffer> buf_gs = get_registered_buffer(ctx, scales_gate,  scale_bytes,  off_gs);
        id<MTLBuffer> buf_up = get_registered_buffer(ctx, packed_up,    packed_bytes, off_up);
        id<MTLBuffer> buf_us = get_registered_buffer(ctx, scales_up,    scale_bytes,  off_us);
        id<MTLBuffer> buf_dp = get_registered_buffer(ctx, packed_down,  packed_bytes, off_dp);
        id<MTLBuffer> buf_ds = get_registered_buffer(ctx, scales_down,  scale_bytes,  off_ds);

        id<MTLBuffer> buf_in = ctx->input_buf;
        id<MTLBuffer> buf_hd = ctx->hidden_buf;
        id<MTLBuffer> buf_ou = ctx->output_buf;
        
        NSUInteger hd_offset = expert_index * ctx->max_rows * sizeof(float);
        NSUInteger ou_offset = expert_index * ctx->max_rows * sizeof(float);

        std::memcpy([buf_in contents], input_vec, input_bytes);

        // ── Encode and commit ─────────────────────────────────────
        id<MTLCommandBuffer>         cmd  = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc  = [cmd computeCommandEncoder];

        // ── Pass 1: smoe_gate_up ──────────────────────────────────
        [enc setComputePipelineState:ctx->gate_up_pso];
        [enc setBuffer:buf_gp offset:off_gp atIndex:0];
        [enc setBuffer:buf_gs offset:off_gs atIndex:1];
        [enc setBuffer:buf_up offset:off_up atIndex:2];
        [enc setBuffer:buf_us offset:off_us atIndex:3];
        [enc setBuffer:buf_in offset:0      atIndex:6];
        [enc setBuffer:buf_hd offset:hd_offset atIndex:7];
        // setBytes: the params struct is 24 bytes — inlined into the
        // command stream. The old newBufferWithBytes here was a heap
        // allocation per dispatch, in the hot loop.
        [enc setBytes:&params length:sizeof(params) atIndex:9];
    // Threadgroup SRAM: the WHOLE input vector, staged once per
    // threadgroup (simdgroup-per-row contract; 16 KB at d_model 4096,
    // under the 32 KB limit up to cols 8192).
    [enc setThreadgroupMemoryLength:gate_cols * sizeof(float) atIndex:0];

    {
        // One SIMDGROUP per output row: 256 threads = 8 rows per
        // threadgroup → ceil(rows/8) threadgroups (192 at ffn_dim 1536;
        // the old thread-per-row grid was 6 — occupancy-starved AND
        // uncoalesced: 22.6 vs 158 GB/s, scratch/kernel_bench_results.md).
        NSUInteger tgroup = 256;
        MTLSize tgSize    = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridGroups  = MTLSizeMake((gate_rows + 7) / 8, 1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // Explicit barrier to ensure gate_up finishes writing to buf_hd before down reads it
    if (@available(macOS 10.14, *)) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else {
        [enc endEncoding];
        enc = [cmd computeCommandEncoder];
    }

    // ── Pass 2: smoe_down ─────────────────────────────────────
    [enc setComputePipelineState:ctx->down_pso];
    [enc setBuffer:buf_dp offset:off_dp atIndex:4];
    [enc setBuffer:buf_ds offset:off_ds atIndex:5];
    [enc setBuffer:buf_hd offset:hd_offset atIndex:7];
    [enc setBuffer:buf_ou offset:ou_offset atIndex:8];
    [enc setBytes:&params length:sizeof(params) atIndex:9];
    [enc setThreadgroupMemoryLength:down_cols * sizeof(float) atIndex:0];

    {
        NSUInteger tgroup = 256;
        MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridGroups = MTLSizeMake((down_rows + 7) / 8, 1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [enc endEncoding];

    // Asynchronous commit - use completion handler to signal when done
    __block uint64_t local_dispatch_count = 0;

    [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // This is called asynchronously when the command buffer completes
        local_dispatch_count = ctx->dispatch_count.fetch_add(1, std::memory_order_relaxed);
    }];

    ctx->signaled_value++;
    uint64_t wait_val = ctx->signaled_value;
    [cmd encodeSignalEvent:ctx->shared_event value:wait_val];

    [cmd commit];

    return (void*)(uintptr_t)wait_val;

    } // end @autoreleasepool
}

void smoe_metal_wait(SmoeMetalCtx* ctx, void* handle, float* output_vec, uint32_t expert_index, uint32_t cols) {
    if (!ctx || !handle) return;
    uint64_t val = (uint64_t)(uintptr_t)handle;
    while (ctx->shared_event.signaledValue < val) {
        std::this_thread::yield();
    }
    std::memcpy(output_vec, ((float*)[ctx->output_buf contents]) + expert_index * ctx->max_rows, cols * sizeof(float));
}

void* smoe_metal_fused_ffn_group(SmoeMetalCtx*        ctx,
                                 const SmoeExpertBlob* experts,
                                 uint32_t              n_experts,
                                 const float*          input_vec,
                                 uint32_t              gate_rows,
                                 uint32_t              gate_cols,
                                 uint32_t              down_rows,
                                 uint32_t              down_cols,
                                 uint32_t              group_size,
                                 uint32_t              bits)
{
    if (!ctx || n_experts == 0) return nullptr;

    struct FusedFFNParamsC { uint32_t gate_rows, gate_cols, down_rows, down_cols, group_size, bits; };
    FusedFFNParamsC params { gate_rows, gate_cols, down_rows, down_cols, group_size, bits };

    @autoreleasepool {
        const uint64_t packed_bytes = static_cast<uint64_t>(gate_rows) * gate_cols * bits / 8;
        const uint64_t scale_bytes  = static_cast<uint64_t>(gate_rows) * gate_cols / group_size * sizeof(uint16_t);
        const uint64_t input_bytes  = static_cast<uint64_t>(gate_cols) * sizeof(float);

        // One input copy for the whole group — every expert of a layer
        // consumes the same normed hidden state.
        std::memcpy([ctx->input_buf contents], input_vec, input_bytes);

        id<MTLCommandBuffer>         cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Simdgroup-per-row contract: 256 threads = 8 rows per
        // threadgroup, grid ceil(rows/8), threadgroup memory holds the
        // whole staged input vector (see kernel header in kMetalSource).
        const NSUInteger tgroup = 256;
        const MTLSize tgSize    = MTLSizeMake(tgroup, 1, 1);
        const MTLSize gridGate  = MTLSizeMake((gate_rows + 7) / 8, 1, 1);
        const MTLSize gridDown  = MTLSizeMake((down_rows + 7) / 8, 1, 1);

        // ── Pass 1: smoe_gate_up for every expert ─────────────────
        // Shape/params/input are identical across the group — bound
        // once. Only the weight buffers and the per-expert hidden
        // region change between dispatches.
        [enc setComputePipelineState:ctx->gate_up_pso];
        [enc setBuffer:ctx->input_buf offset:0 atIndex:6];
        [enc setBytes:&params length:sizeof(params) atIndex:9];
        [enc setThreadgroupMemoryLength:gate_cols * sizeof(float) atIndex:0];

        for (uint32_t i = 0; i < n_experts; ++i) {
            NSUInteger off_gp = 0, off_gs = 0, off_up = 0, off_us = 0;
            id<MTLBuffer> buf_gp = get_registered_buffer(ctx, experts[i].packed_gate, packed_bytes, off_gp);
            id<MTLBuffer> buf_gs = get_registered_buffer(ctx, experts[i].scales_gate, scale_bytes,  off_gs);
            id<MTLBuffer> buf_up = get_registered_buffer(ctx, experts[i].packed_up,   packed_bytes, off_up);
            id<MTLBuffer> buf_us = get_registered_buffer(ctx, experts[i].scales_up,   scale_bytes,  off_us);

            [enc setBuffer:buf_gp offset:off_gp atIndex:0];
            [enc setBuffer:buf_gs offset:off_gs atIndex:1];
            [enc setBuffer:buf_up offset:off_up atIndex:2];
            [enc setBuffer:buf_us offset:off_us atIndex:3];
            [enc setBuffer:ctx->hidden_buf
                   offset:experts[i].slot * ctx->max_rows * sizeof(float)
                  atIndex:7];
            [enc dispatchThreadgroups:gridGate threadsPerThreadgroup:tgSize];
        }

        // One barrier for the whole group: every down pass reads only
        // its own expert's hidden region, written by pass 1 above.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ── Pass 2: smoe_down for every expert ────────────────────
        [enc setComputePipelineState:ctx->down_pso];
        [enc setBytes:&params length:sizeof(params) atIndex:9];
        [enc setThreadgroupMemoryLength:down_cols * sizeof(float) atIndex:0];

        for (uint32_t i = 0; i < n_experts; ++i) {
            NSUInteger off_dp = 0, off_ds = 0;
            id<MTLBuffer> buf_dp = get_registered_buffer(ctx, experts[i].packed_down, packed_bytes, off_dp);
            id<MTLBuffer> buf_ds = get_registered_buffer(ctx, experts[i].scales_down, scale_bytes,  off_ds);

            const NSUInteger region = experts[i].slot * ctx->max_rows * sizeof(float);
            [enc setBuffer:buf_dp offset:off_dp atIndex:4];
            [enc setBuffer:buf_ds offset:off_ds atIndex:5];
            [enc setBuffer:ctx->hidden_buf offset:region atIndex:7];
            [enc setBuffer:ctx->output_buf offset:region atIndex:8];
            [enc dispatchThreadgroups:gridDown threadsPerThreadgroup:tgSize];
        }

        [enc endEncoding];

        [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            (void)buffer;
            ctx->dispatch_count.fetch_add(n_experts, std::memory_order_relaxed);
        }];

        ctx->signaled_value++;
        uint64_t wait_val = ctx->signaled_value;
        [cmd encodeSignalEvent:ctx->shared_event value:wait_val];

        [cmd commit];

        return (void*)(uintptr_t)wait_val;
    } // end @autoreleasepool
}

void* smoe_metal_fused_ffn_batch(SmoeMetalCtx*   ctx,
                                 const uint8_t*  packed_gate,
                                 const uint16_t* scales_gate,
                                 const uint8_t*  packed_up,
                                 const uint16_t* scales_up,
                                 const uint8_t*  packed_down,
                                 const uint16_t* scales_down,
                                 const float*    input_mat,
                                 uint32_t        batch,
                                 uint32_t        gate_rows,
                                 uint32_t        gate_cols,
                                 uint32_t        down_rows,
                                 uint32_t        down_cols,
                                 uint32_t        group_size,
                                 uint32_t        bits)
{
    if (!ctx || !ctx->gate_up_batch_pso || batch == 0 ||
        batch > SMOE_MAX_FFN_BATCH ||
        gate_rows > SMOE_MAX_FFN_BATCH_DIM || gate_cols > SMOE_MAX_FFN_BATCH_DIM ||
        down_rows > SMOE_MAX_FFN_BATCH_DIM || down_cols > SMOE_MAX_FFN_BATCH_DIM) {
        return nullptr; // caller falls back to the per-token path
    }

    struct FusedFFNBatchParamsC {
        uint32_t gate_rows, gate_cols, down_rows, down_cols, group_size, bits, batch;
    };
    FusedFFNBatchParamsC params { gate_rows, gate_cols, down_rows, down_cols,
                                  group_size, bits, batch };

    @autoreleasepool {
        uint64_t packed_bytes = static_cast<uint64_t>(gate_rows) * gate_cols * bits / 8;
        uint64_t scale_bytes  = static_cast<uint64_t>(gate_rows) * gate_cols / group_size * sizeof(uint16_t);

        NSUInteger off_gp = 0, off_gs = 0, off_up = 0, off_us = 0, off_dp = 0, off_ds = 0;
        id<MTLBuffer> buf_gp = get_registered_buffer(ctx, packed_gate,  packed_bytes, off_gp);
        id<MTLBuffer> buf_gs = get_registered_buffer(ctx, scales_gate,  scale_bytes,  off_gs);
        id<MTLBuffer> buf_up = get_registered_buffer(ctx, packed_up,    packed_bytes, off_up);
        id<MTLBuffer> buf_us = get_registered_buffer(ctx, scales_up,    scale_bytes,  off_us);
        id<MTLBuffer> buf_dp = get_registered_buffer(ctx, packed_down,  packed_bytes, off_dp);
        id<MTLBuffer> buf_ds = get_registered_buffer(ctx, scales_down,  scale_bytes,  off_ds);

        std::memcpy([ctx->batch_in_buf contents], input_mat,
                    static_cast<size_t>(batch) * gate_cols * sizeof(float));

        id<MTLCommandBuffer>         cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Pass 1: gate+up+SiLU for all batch rows of this expert
        [enc setComputePipelineState:ctx->gate_up_batch_pso];
        [enc setBuffer:buf_gp offset:off_gp atIndex:0];
        [enc setBuffer:buf_gs offset:off_gs atIndex:1];
        [enc setBuffer:buf_up offset:off_up atIndex:2];
        [enc setBuffer:buf_us offset:off_us atIndex:3];
        [enc setBuffer:ctx->batch_in_buf     offset:0 atIndex:6];
        [enc setBuffer:ctx->batch_hidden_buf offset:0 atIndex:7];
        [enc setBytes:&params length:sizeof(params) atIndex:9];
        [enc setThreadgroupMemoryLength:256 * 2 * sizeof(float) atIndex:0];
        {
            MTLSize tg   = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((gate_rows + 255) / 256, batch, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Pass 2: down projection for all batch rows
        [enc setComputePipelineState:ctx->down_batch_pso];
        [enc setBuffer:buf_dp offset:off_dp atIndex:4];
        [enc setBuffer:buf_ds offset:off_ds atIndex:5];
        [enc setBuffer:ctx->batch_hidden_buf offset:0 atIndex:7];
        [enc setBuffer:ctx->batch_out_buf    offset:0 atIndex:8];
        [enc setBytes:&params length:sizeof(params) atIndex:9];
        [enc setThreadgroupMemoryLength:256 * 2 * sizeof(float) atIndex:0];
        {
            MTLSize tg   = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake((down_rows + 255) / 256, batch, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];

        [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            ctx->dispatch_count.fetch_add(1, std::memory_order_relaxed);
        }];

        ctx->signaled_value++;
        uint64_t wait_val = ctx->signaled_value;
        [cmd encodeSignalEvent:ctx->shared_event value:wait_val];
        [cmd commit];

        return (void*)(uintptr_t)wait_val;
    }
}

void smoe_metal_wait_ffn_batch(SmoeMetalCtx* ctx, void* handle,
                               float* out_mat, uint32_t batch, uint32_t cols) {
    if (!ctx || !handle) return;
    uint64_t val = (uint64_t)(uintptr_t)handle;
    while (ctx->shared_event.signaledValue < val) {
        std::this_thread::yield();
    }
    std::memcpy(out_mat, [ctx->batch_out_buf contents],
                static_cast<size_t>(batch) * cols * sizeof(float));
}

void smoe_metal_swap_buffers(SmoeMetalCtx* ctx) {
    if (!ctx) return;
    // Plain ObjC reference swap — no locks.
    // The caller (main loop) ensures this is called only when
    // buf_a has been fully consumed.
    std::swap(ctx->buf_a, ctx->buf_b);
    // Signal to the I/O streamer that the new passive buffer is ready.
    ctx->passive_ready.store(true, std::memory_order_release);
}

bool smoe_metal_passive_ready(const SmoeMetalCtx* ctx) {
    if (!ctx) return false;
    return ctx->passive_ready.load(std::memory_order_acquire);
}

uint64_t smoe_metal_kernel_dispatches(const SmoeMetalCtx* ctx) {
    if (!ctx) return 0;
    return ctx->dispatch_count.load(std::memory_order_relaxed);
}

void smoe_metal_scout_matvec(SmoeMetalCtx* ctx,
                             const float*   weight,
                             const float*   input_vec,
                             float*         output_vec,
                             uint32_t       rows,
                             uint32_t       cols)
{
    if (!ctx) return;

    @autoreleasepool {
    MTLResourceOptions uma = MTLResourceStorageModeShared;

    size_t weight_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    size_t input_bytes  = static_cast<size_t>(cols) * sizeof(float);
    size_t output_bytes = static_cast<size_t>(rows) * sizeof(float);

    NSUInteger off_wt = 0, off_in = 0, off_ou = 0;
    id<MTLBuffer> buf_wt = get_registered_buffer(ctx, weight,     weight_bytes, off_wt);
    id<MTLBuffer> buf_in = get_registered_buffer(ctx, input_vec,  input_bytes,  off_in);
    id<MTLBuffer> buf_ou = get_registered_buffer(ctx, output_vec, output_bytes, off_ou);

    if (!buf_wt || !buf_in || !buf_ou) {
        std::fprintf(stderr, "[smoe_metal] ERROR: failed to wrap matvec pointers.\n");
        return;
    }

    struct Dims { uint32_t rows, cols; };
    Dims dims { rows, cols };

    id<MTLCommandBuffer>         cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:ctx->scout_matvec_pso];
    [enc setBuffer:buf_wt offset:off_wt atIndex:0];
    [enc setBuffer:buf_in offset:off_in atIndex:1];
    [enc setBuffer:buf_ou offset:off_ou atIndex:2];
    [enc setBytes:&dims length:sizeof(dims) atIndex:3];

    [enc setThreadgroupMemoryLength:cols * sizeof(float) atIndex:0];

    NSUInteger tgroup = ctx->scout_matvec_pso.maxTotalThreadsPerThreadgroup;
    tgroup = std::min(tgroup, NSUInteger(256));
    MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
    MTLSize gridSize = MTLSizeMake(rows,   1, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    [enc endEncoding];

    ctx->signaled_value++;
    uint64_t wait_val = ctx->signaled_value;
    [cmd encodeSignalEvent:ctx->shared_event value:wait_val];

    [cmd commit];

    while (ctx->shared_event.signaledValue < wait_val) {
        std::this_thread::yield();
    }
    }
}

void smoe_metal_scout_matvec_batch(SmoeMetalCtx* ctx,
                                   const float**  weights,
                                   const float**  inputs,
                                   float**        outputs,
                                   const uint32_t* rows,
                                   const uint32_t* cols,
                                   uint32_t       count)
{
    if (!ctx || count == 0) return;

    @autoreleasepool {
    MTLResourceOptions uma = MTLResourceStorageModeShared;

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx->scout_matvec_pso];

    for (uint32_t idx = 0; idx < count; ++idx) {
        uint32_t r = rows[idx];
        uint32_t c = cols[idx];

        size_t weight_bytes = static_cast<size_t>(r) * c * sizeof(float);
        size_t input_bytes  = static_cast<size_t>(c) * sizeof(float);
        size_t output_bytes = static_cast<size_t>(r) * sizeof(float);

        NSUInteger off_wt = 0, off_in = 0, off_ou = 0;
        id<MTLBuffer> buf_wt = get_registered_buffer(ctx, weights[idx], weight_bytes, off_wt);
        id<MTLBuffer> buf_in = get_registered_buffer(ctx, inputs[idx],  input_bytes,  off_in);
        id<MTLBuffer> buf_ou = get_registered_buffer(ctx, outputs[idx], output_bytes, off_ou);

        if (!buf_wt || !buf_in || !buf_ou) {
            std::fprintf(stderr, "[smoe_metal] ERROR: failed to wrap pointers in batch index %u.\n", idx);
            continue;
        }

        struct Dims { uint32_t rows, cols; };
        Dims dims { r, c };

        [enc setBuffer:buf_wt offset:off_wt atIndex:0];
        [enc setBuffer:buf_in offset:off_in atIndex:1];
        [enc setBuffer:buf_ou offset:off_ou atIndex:2];
        [enc setBytes:&dims length:sizeof(dims) atIndex:3];

        [enc setThreadgroupMemoryLength:c * sizeof(float) atIndex:0];

        NSUInteger tgroup = ctx->scout_matvec_pso.maxTotalThreadsPerThreadgroup;
        tgroup = std::min(tgroup, NSUInteger(256));
        MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(r,      1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }

    [enc endEncoding];
    ctx->signaled_value++;
    uint64_t wait_val = ctx->signaled_value;
    [cmd encodeSignalEvent:ctx->shared_event value:wait_val];

    [cmd commit];

    while (ctx->shared_event.signaledValue < wait_val) {
        std::this_thread::yield();
    }

    ctx->dispatch_count.fetch_add(count, std::memory_order_relaxed);
    }
}

void smoe_metal_register_buffer(SmoeMetalCtx* ctx, const void* ptr, size_t size_in_bytes) {
    if (!ctx || !ptr || size_in_bytes == 0) return;

    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& reg : ctx->registered_buffers) {
        uintptr_t base = reinterpret_cast<uintptr_t>(reg.ptr);
        if (p >= base && (p + size_in_bytes) <= (base + reg.length)) return;
    }

    size_t aligned = (size_in_bytes + (smoe::PAGE_SIZE - 1)) & ~(smoe::PAGE_SIZE - 1);
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                                       length:aligned
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (buf) {
        ctx->registered_buffers.push_back({ptr, aligned, buf});
    } else {
        std::fprintf(stderr, "[smoe_metal] ERROR: failed to register buffer at %p (%zu bytes)\n", ptr, size_in_bytes);
    }
}



void smoe_metal_scout_matvec_batch_bf16(SmoeMetalCtx* ctx,
                                   const uint16_t** weights,
                                   const float**  inputs,
                                   float**        outputs,
                                   const uint32_t* rows,
                                   const uint32_t* cols,
                                   uint32_t       count)
{
    if (!ctx || count == 0) return;

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:ctx->scout_matvec_bf16_pso];

    for (uint32_t i = 0; i < count; ++i) {
        size_t weight_bytes = static_cast<size_t>(rows[i]) * cols[i] * sizeof(uint16_t);
        size_t input_bytes  = static_cast<size_t>(cols[i]) * sizeof(float);
        size_t output_bytes = static_cast<size_t>(rows[i]) * sizeof(float);

        NSUInteger off_wt=0, off_in=0, off_ou=0;
        id<MTLBuffer> buf_wt = get_registered_buffer(ctx, weights[i], weight_bytes, off_wt);
        id<MTLBuffer> buf_in = get_registered_buffer(ctx, inputs[i],  input_bytes,  off_in);
        id<MTLBuffer> buf_ou = get_registered_buffer(ctx, outputs[i], output_bytes, off_ou);

        if (!buf_wt || !buf_in || !buf_ou) {
            std::fprintf(stderr, "[smoe_metal] ERROR: failed to wrap batch bf16 pointers.\n");
            return;
        }

        Dims dims { rows[i], cols[i] };

        [enc setBuffer:buf_wt offset:off_wt atIndex:0];
        [enc setBuffer:buf_in offset:off_in atIndex:1];
        [enc setBuffer:buf_ou offset:off_ou atIndex:2];
        [enc setBytes:&dims length:sizeof(dims) atIndex:3];

        NSUInteger tgroup = 256;
        [enc setThreadgroupMemoryLength:tgroup * sizeof(float) atIndex:0];

        MTLSize tgSize     = MTLSizeMake(tgroup, 1, 1);
        // ceil(rows/256) threadgroups, one thread per row. This path runs
        // the 151936-row LM head once per generated token — the old
        // one-threadgroup-per-row grid was a 256× overdispatch.
        MTLSize gridGroups = MTLSizeMake((rows[i] + tgroup - 1) / tgroup, 1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [enc endEncoding];
    
    ctx->signaled_value++;
    uint64_t wait_val = ctx->signaled_value;
    [cmd encodeSignalEvent:ctx->shared_event value:wait_val];

    [cmd commit];

    while (ctx->shared_event.signaledValue < wait_val) {
        std::this_thread::yield();
    }
    }
}

void smoe_metal_scout_matvec_bf16(SmoeMetalCtx* ctx,
                             const uint16_t* weight,
                             const float*  input_vec,
                             float*        output_vec,
                             uint32_t      rows,
                             uint32_t      cols)
{
    const uint16_t* weights[1] = { weight };
    const float* inputs[1] = { input_vec };
    float* outputs[1] = { output_vec };
    uint32_t r[1] = { rows };
    uint32_t c[1] = { cols };
    smoe_metal_scout_matvec_batch_bf16(ctx, weights, inputs, outputs, r, c, 1);
}

// Host-side mirrors of the MSL param structs (4-byte fields only —
// identical layout on both sides, safe for setBytes).
struct AttnPrepParamsHost {
    uint32_t head_dim, num_heads, num_kv_heads, rope_half;
    uint32_t slot, kv_dim, has_qk_norm;
};
struct AttnDecodeParamsHost {
    uint32_t head_dim, num_heads, heads_per_kv, kv_dim;
    uint32_t slot, valid, attn_ctx;
    float    scale;
};

void smoe_metal_attention_layer(SmoeMetalCtx* ctx, const SmoeAttnLayerArgs* a)
{
    if (!ctx || !a) return;

    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    const size_t in_bytes  = size_t(a->d_model) * sizeof(float);
    const size_t cache_bytes = size_t(a->attn_ctx) * a->kv_dim * sizeof(float);

    // ── Pass 1: Q/K/V projections (independent — one barrier after) ──
    [enc setComputePipelineState:ctx->scout_matvec_bf16_pso];
    {
        const uint16_t* w[3]   = { a->w_q,  a->w_k,  a->w_v  };
        float*          o[3]   = { a->qbuf, a->kbuf, a->vbuf };
        const uint32_t  rows[3]= { a->q_dim, a->kv_dim, a->kv_dim };

        NSUInteger off_in = 0;
        id<MTLBuffer> buf_in = get_registered_buffer(ctx, a->normed_in, in_bytes, off_in);
        for (int i = 0; i < 3; ++i) {
            size_t w_bytes = size_t(rows[i]) * a->d_model * sizeof(uint16_t);
            NSUInteger off_wt = 0, off_ou = 0;
            id<MTLBuffer> buf_wt = get_registered_buffer(ctx, w[i], w_bytes, off_wt);
            id<MTLBuffer> buf_ou = get_registered_buffer(ctx, o[i], size_t(rows[i]) * sizeof(float), off_ou);
            if (!buf_wt || !buf_in || !buf_ou) {
                std::fprintf(stderr, "[smoe_metal] ERROR: attention QKV wrap failed.\n");
                [enc endEncoding];
                return;
            }
            Dims dims { rows[i], a->d_model };
            [enc setBuffer:buf_wt offset:off_wt atIndex:0];
            [enc setBuffer:buf_in offset:off_in atIndex:1];
            [enc setBuffer:buf_ou offset:off_ou atIndex:2];
            [enc setBytes:&dims length:sizeof(dims) atIndex:3];
            NSUInteger tgroup = 256;
            [enc setThreadgroupMemoryLength:tgroup * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((rows[i] + tgroup - 1) / tgroup, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tgroup, 1, 1)];
        }
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ── Pass 2: per-head QK norm + RoPE + KV ring append ──
    {
        NSUInteger off_q = 0, off_k = 0, off_v = 0, off_qn = 0, off_kn = 0,
                   off_c = 0, off_s = 0, off_kc = 0, off_vc = 0;
        size_t head_bytes = size_t(a->head_dim) * sizeof(uint16_t);
        size_t rope_bytes = size_t(a->head_dim / 2) * sizeof(float);
        id<MTLBuffer> buf_q  = get_registered_buffer(ctx, a->qbuf, size_t(a->q_dim) * sizeof(float), off_q);
        id<MTLBuffer> buf_k  = get_registered_buffer(ctx, a->kbuf, size_t(a->kv_dim) * sizeof(float), off_k);
        id<MTLBuffer> buf_v  = get_registered_buffer(ctx, a->vbuf, size_t(a->kv_dim) * sizeof(float), off_v);
        // QK-norm weights may be absent: bind qbuf as a placeholder so
        // the slots are valid; the kernel never reads them then.
        id<MTLBuffer> buf_qn = a->q_norm_w ? get_registered_buffer(ctx, a->q_norm_w, head_bytes, off_qn) : buf_q;
        id<MTLBuffer> buf_kn = a->k_norm_w ? get_registered_buffer(ctx, a->k_norm_w, head_bytes, off_kn) : buf_q;
        id<MTLBuffer> buf_c  = get_registered_buffer(ctx, a->rope_cos, rope_bytes, off_c);
        id<MTLBuffer> buf_s  = get_registered_buffer(ctx, a->rope_sin, rope_bytes, off_s);
        id<MTLBuffer> buf_kc = get_registered_buffer(ctx, a->k_cache, cache_bytes, off_kc);
        id<MTLBuffer> buf_vc = get_registered_buffer(ctx, a->v_cache, cache_bytes, off_vc);
        if (!buf_q || !buf_k || !buf_v || !buf_qn || !buf_kn || !buf_c || !buf_s || !buf_kc || !buf_vc) {
            std::fprintf(stderr, "[smoe_metal] ERROR: attention prep wrap failed.\n");
            [enc endEncoding];
            return;
        }
        AttnPrepParamsHost pp {
            a->head_dim, a->num_heads, a->num_kv_heads, a->head_dim / 2,
            a->slot, a->kv_dim,
            (a->q_norm_w && a->k_norm_w) ? 1u : 0u
        };
        [enc setComputePipelineState:ctx->attn_prep_pso];
        [enc setBuffer:buf_q  offset:off_q  atIndex:0];
        [enc setBuffer:buf_k  offset:off_k  atIndex:1];
        [enc setBuffer:buf_v  offset:off_v  atIndex:2];
        [enc setBuffer:buf_qn offset:(a->q_norm_w ? off_qn : off_q) atIndex:3];
        [enc setBuffer:buf_kn offset:(a->k_norm_w ? off_kn : off_q) atIndex:4];
        [enc setBuffer:buf_c  offset:off_c  atIndex:5];
        [enc setBuffer:buf_s  offset:off_s  atIndex:6];
        [enc setBuffer:buf_kc offset:off_kc atIndex:7];
        [enc setBuffer:buf_vc offset:off_vc atIndex:8];
        [enc setBytes:&pp length:sizeof(pp) atIndex:9];
        [enc setThreadgroupMemoryLength:(a->head_dim + 32) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(a->num_heads + a->num_kv_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(a->head_dim, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ── Pass 3: GQA attention over the KV ring ──
    {
        NSUInteger off_q = 0, off_kc = 0, off_vc = 0, off_o = 0;
        id<MTLBuffer> buf_q  = get_registered_buffer(ctx, a->qbuf, size_t(a->q_dim) * sizeof(float), off_q);
        id<MTLBuffer> buf_kc = get_registered_buffer(ctx, a->k_cache, cache_bytes, off_kc);
        id<MTLBuffer> buf_vc = get_registered_buffer(ctx, a->v_cache, cache_bytes, off_vc);
        id<MTLBuffer> buf_o  = get_registered_buffer(ctx, a->attn_out, size_t(a->q_dim) * sizeof(float), off_o);
        if (!buf_q || !buf_kc || !buf_vc || !buf_o) {
            std::fprintf(stderr, "[smoe_metal] ERROR: attention decode wrap failed.\n");
            [enc endEncoding];
            return;
        }
        AttnDecodeParamsHost dp {
            a->head_dim, a->num_heads, a->num_heads / a->num_kv_heads, a->kv_dim,
            a->slot, a->valid, a->attn_ctx,
            a->attn_scale
        };
        [enc setComputePipelineState:ctx->attn_decode_pso];
        [enc setBuffer:buf_q  offset:off_q  atIndex:0];
        [enc setBuffer:buf_kc offset:off_kc atIndex:1];
        [enc setBuffer:buf_vc offset:off_vc atIndex:2];
        [enc setBuffer:buf_o  offset:off_o  atIndex:3];
        [enc setBytes:&dp length:sizeof(dp) atIndex:4];
        // Threadgroup plan: staged Q head + score row + simd scratch.
        [enc setThreadgroupMemoryLength:(a->head_dim + a->valid + 32) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(a->num_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ── Pass 4: O projection ──
    {
        size_t w_bytes = size_t(a->d_model) * a->q_dim * sizeof(uint16_t);
        NSUInteger off_wt = 0, off_in = 0, off_ou = 0;
        id<MTLBuffer> buf_wt = get_registered_buffer(ctx, a->w_o, w_bytes, off_wt);
        id<MTLBuffer> buf_in = get_registered_buffer(ctx, a->attn_out, size_t(a->q_dim) * sizeof(float), off_in);
        id<MTLBuffer> buf_ou = get_registered_buffer(ctx, a->o_out, size_t(a->d_model) * sizeof(float), off_ou);
        if (!buf_wt || !buf_in || !buf_ou) {
            std::fprintf(stderr, "[smoe_metal] ERROR: attention o_proj wrap failed.\n");
            [enc endEncoding];
            return;
        }
        Dims dims { a->d_model, a->q_dim };
        [enc setComputePipelineState:ctx->scout_matvec_bf16_pso];
        [enc setBuffer:buf_wt offset:off_wt atIndex:0];
        [enc setBuffer:buf_in offset:off_in atIndex:1];
        [enc setBuffer:buf_ou offset:off_ou atIndex:2];
        [enc setBytes:&dims length:sizeof(dims) atIndex:3];
        NSUInteger tgroup = 256;
        [enc setThreadgroupMemoryLength:tgroup * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((a->d_model + tgroup - 1) / tgroup, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgroup, 1, 1)];
    }

    [enc endEncoding];

    ctx->signaled_value++;
    uint64_t wait_val = ctx->signaled_value;
    [cmd encodeSignalEvent:ctx->shared_event value:wait_val];
    [cmd commit];

    while (ctx->shared_event.signaledValue < wait_val) {
        std::this_thread::yield();
    }
    }
}
