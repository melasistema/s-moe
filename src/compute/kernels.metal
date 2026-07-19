// ═══════════════════════════════════════════════════════════════
// kernels.metal — S-MoE Engine · SMOE-Q2 Fused FFN Compute Shader
// ═══════════════════════════════════════════════════════════════
// Single source of truth for the GPU kernels: make embeds this file
// verbatim as build/kernels_msl.h and the bridge JIT-compiles it at
// boot (the binary stays self-contained, no .metallib to ship).
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

// ── Scout BF16 Matvec (simdgroup-per-row) ─────────────────────
// Coalesced twin of scout_matvec_bf16, same dispatch contract as
// the fused FFN kernels above:
//   grid            : ceil(rows / ROWS_PER_TG) threadgroups
//   threadsPerTG    : TGROUP_SIZE (256 = 8 simdgroups × 32 lanes)
//   threadgroup mem : cols × sizeof(float) — the WHOLE staged input
// One simdgroup owns one output row; its 32 lanes read consecutive
// uint4 chunks (8 bf16 each — 512 coalesced bytes per step) instead
// of one ushort each from addresses cols×2 bytes apart.
// Contract (bridge-checked per dispatch; thread-per-row twin covers
// the rest):
//   • weight pointer 16-byte aligned and cols % 8 == 0, so every
//     row base stays 16-byte aligned for the uint4 loads
//   • cols × 4 ≤ maxThreadgroupMemoryLength (32 KB → cols ≤ 8192)
kernel void scout_matvec_bf16_sg(
    device const uint16_t* weights [[buffer(0)]],
    device const float*    input   [[buffer(1)]],
    device       float*    output  [[buffer(2)]],
    constant     Dims&     dims    [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* xshared [[threadgroup(0)]])
{
    // Stage the input vector once per threadgroup (shared by 8 rows).
    for (uint i = tid; i < dims.cols; i += TGROUP_SIZE) {
        xshared[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tgid * ROWS_PER_TG + sgi;
    if (row >= dims.rows) return;   // whole simdgroup exits together

    device const uint4* rp =
        (device const uint4*)(weights + (ulong)row * dims.cols);
    const uint nchunks = dims.cols >> 3;   // 8 bf16 per 16-byte chunk

    float4 acc = float4(0.0f);
    for (uint c = lane; c < nchunks; c += 32) {
        uint4 v = rp[c];
        threadgroup const float4* xv =
            (threadgroup const float4*)(xshared + (c << 3));
        // Two bf16 per uint: the low half needs the <<16 shift, the
        // high half is already in sign/exponent position — mask only.
        float4 w0 = float4(as_type<float>(v.x << 16),
                           as_type<float>(v.x & 0xFFFF0000u),
                           as_type<float>(v.y << 16),
                           as_type<float>(v.y & 0xFFFF0000u));
        float4 w1 = float4(as_type<float>(v.z << 16),
                           as_type<float>(v.z & 0xFFFF0000u),
                           as_type<float>(v.w << 16),
                           as_type<float>(v.w & 0xFFFF0000u));
        acc = fma(w0, xv[0], acc);
        acc = fma(w1, xv[1], acc);
    }
    float r = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) output[row] = r;
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

// ═══════════════════════════════════════════════════════════════
// Token-batch fused FFN (layer-major prefill)
// ═══════════════════════════════════════════════════════════════
// One dispatch applies one expert to `batch` token activations:
// grid = (ceil(rows/256), batch). Row-major matrices:
//   input  [batch × gate_cols]   hidden [batch × gate_rows]
//   output [batch × down_rows]

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
