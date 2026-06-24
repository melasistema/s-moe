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
    return (float(code) - 1.5f) * scale;
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
    uint                       row          [[threadgroup_position_in_grid]],
    uint                       tid          [[thread_index_in_threadgroup]],
    uint                       threads_per_tg [[threads_per_threadgroup]],
    threadgroup float*         tg_acc       [[threadgroup(0)]])
{
    if (row >= params.rows) return;
    
    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    uint wi_base = row * params.cols;
    
    if (params.bits == 4) {
        // 2 weights per byte
        uint bytes_per_row = params.cols / 2;
        uint byte_base = wi_base / 2;
        
        for (uint i = tid; i < bytes_per_row; i += threads_per_tg) {
            uint pidx = byte_base + i;
            uint8_t gbyte = gate_packed[pidx];
            uint8_t ubyte = up_packed[pidx];
            
            uint k = i * 2;
            float gs = float(gate_scales[(wi_base + k) / params.group_size]);
            float us = float(up_scales[(wi_base + k) / params.group_size]);
            
            for (uint b = 0; b < 2; ++b) {
                if (k + b < params.cols) {
                    float gcode = float((gbyte >> (b * 4)) & 0xFu);
                    float ucode = float((ubyte >> (b * 4)) & 0xFu);
                    float gw = (gcode - 8.0f) * gs;
                    float uw = (ucode - 8.0f) * us;
                    float x_k = input[k + b];
                    gate_acc += gw * x_k;
                    up_acc   += uw * x_k;
                }
            }
        }
    } else {
        // 4 weights per byte
        uint bytes_per_row = params.cols / 4;
        uint byte_base = wi_base / 4;
        
        for (uint i = tid; i < bytes_per_row; i += threads_per_tg) {
            uint pidx = byte_base + i;
            uint8_t gbyte = gate_packed[pidx];
            uint8_t ubyte = up_packed[pidx];
            
            uint k = i * 4;
            float gs = float(gate_scales[(wi_base + k) / params.group_size]);
            float us = float(up_scales[(wi_base + k) / params.group_size]);
            
            for (uint b = 0; b < 4; ++b) {
                if (k + b < params.cols) {
                    float gcode = float((gbyte >> (b * 2)) & 0x3u);
                    float ucode = float((ubyte >> (b * 2)) & 0x3u);
                    float gw = (gcode - 1.5f) * (1.0f / 1.5f) * gs;
                    float uw = (ucode - 1.5f) * (1.0f / 1.5f) * us;
                    float x_k = input[k + b];
                    gate_acc += gw * x_k;
                    up_acc   += uw * x_k;
                }
            }
        }
    }
    
    threadgroup float* tg_gate = tg_acc;
    threadgroup float* tg_up   = tg_acc + threads_per_tg;
    
    tg_gate[tid] = gate_acc;
    tg_up[tid]   = up_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_gate[tid] += tg_gate[tid + stride];
            tg_up[tid]   += tg_up[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        // SiLU fused
        float g = tg_gate[0];
        float u = tg_up[0];
        hidden[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void smoe_down(
    device const uint8_t*      down_packed  [[buffer(4)]],
    device const half*         down_scales  [[buffer(5)]],
    device const float*        hidden       [[buffer(7)]],
    device       float*        output       [[buffer(8)]],
    constant FusedFFNParams&   params       [[buffer(9)]],
    uint                       row          [[threadgroup_position_in_grid]],
    uint                       tid          [[thread_index_in_threadgroup]],
    uint                       threads_per_tg [[threads_per_threadgroup]],
    threadgroup float*         tg_acc       [[threadgroup(0)]])
{
    if (row >= params.cols) return;
    
    float acc = 0.0f;
    uint wi_base = row * params.rows;
    
    if (params.bits == 4) {
        uint bytes_per_row = params.rows / 2;
        uint byte_base = wi_base / 2;
        
        for (uint i = tid; i < bytes_per_row; i += threads_per_tg) {
            uint pidx = byte_base + i;
            uint8_t dbyte = down_packed[pidx];
            
            uint k = i * 2;
            float ds = float(down_scales[(wi_base + k) / params.group_size]);
            
            for (uint b = 0; b < 2; ++b) {
                if (k + b < params.rows) {
                    float dcode = float((dbyte >> (b * 4)) & 0xFu);
                    float dw = (dcode - 8.0f) * ds;
                    acc += dw * hidden[k + b];
                }
            }
        }
    } else {
        uint bytes_per_row = params.rows / 4;
        uint byte_base = wi_base / 4;
        
        for (uint i = tid; i < bytes_per_row; i += threads_per_tg) {
            uint pidx = byte_base + i;
            uint8_t dbyte = down_packed[pidx];
            
            uint k = i * 4;
            float ds = float(down_scales[(wi_base + k) / params.group_size]);
            
            for (uint b = 0; b < 4; ++b) {
                if (k + b < params.rows) {
                    float dcode = float((dbyte >> (b * 2)) & 0x3u);
                    float dw = (dcode - 1.5f) * (1.0f / 1.5f) * ds;
                    acc += dw * hidden[k + b];
                }
            }
        }
    }
    
    tg_acc[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_acc[tid] += tg_acc[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[row] = tg_acc[0];
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




kernel void scout_matvec_bf16(
    device const ushort* weight       [[buffer(0)]],
    device const float*  input_vec    [[buffer(1)]],
    device       float*  output_vec   [[buffer(2)]],
    constant     Dims&   dims         [[buffer(3)]],
    uint                 row          [[threadgroup_position_in_grid]],
    uint                 tid          [[thread_index_in_threadgroup]],
    uint                 threads_per_tg [[threads_per_threadgroup]],
    threadgroup  float*  tg_sum       [[threadgroup(0)]])
{
    uint cols = dims.cols;
    if (row >= dims.rows) return;

    device const ushort* row_w = weight + row * cols;
    
    float acc = 0.0f;
    for (uint i = tid; i < cols; i += threads_per_tg) {
        float w_f = bf16_to_f32(row_w[i]);
        acc += w_f * input_vec[i];
    }
    
    tg_sum[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_sum[tid] += tg_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output_vec[row] = tg_sum[0];
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
    NSString* source = [NSString stringWithUTF8String:kMetalSource];

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
                           uint32_t        rows,
                           uint32_t        cols,
                           uint32_t        group_size,
                           uint32_t        bits,
                           uint32_t        expert_index)
{
    if (!ctx) return nullptr;

    // ── Build FusedFFNParams ──────────────────────────────────
    // Stack allocation — not in hot-path alloc-banned zone because
    // this struct is tiny and compiler-stack-allocated.
    struct FusedFFNParamsC { uint32_t rows, cols, group_size, bits; };
    FusedFFNParamsC params { rows, cols, group_size, bits };

    // ── Wrap raw pointers in MTLBuffers (no-copy) ─────────────
    MTLResourceOptions uma = MTLResourceStorageModeShared;

    @autoreleasepool {
        // Compute byte sizes for each sub-tensor
        uint64_t packed_bytes = static_cast<uint64_t>(rows) * cols * bits / 8;
        uint64_t scale_bytes  = static_cast<uint64_t>(rows) * cols / group_size * sizeof(uint16_t);
        uint64_t input_bytes  = static_cast<uint64_t>(cols)  * sizeof(float);
        uint64_t output_bytes = static_cast<uint64_t>(rows)  * sizeof(float);

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

        // Params as a tiny inline buffer
        id<MTLBuffer> buf_pa = [ctx->device newBufferWithBytes:&params
                                                        length:sizeof(params)
                                                       options:uma];

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
        [enc setBuffer:buf_pa offset:0      atIndex:9];
    // Threadgroup SRAM: TGROUP_SIZE floats for the input tile
    [enc setThreadgroupMemoryLength:256 * 2 * sizeof(float) atIndex:0];

    {
        NSUInteger tgroup = 256;
        MTLSize tgSize    = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridGroups  = MTLSizeMake(rows,   1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
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
    [enc setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];

    {
        NSUInteger tgroup = 256;
        MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridGroups = MTLSizeMake(cols,   1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
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

    // Asynchronous commit - use completion handler to signal when done
    __block bool completed = false;
    __block uint64_t local_dispatch_count = 0;

    [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        // This is called asynchronously when the command buffer completes
        local_dispatch_count = ctx->dispatch_count.fetch_add(1, std::memory_order_relaxed);
        completed = true;
    }];

    [cmd commit];
    [cmd waitUntilCompleted];
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
    [cmd commit];
    [cmd waitUntilCompleted];

    ctx->dispatch_count.fetch_add(count, std::memory_order_relaxed);
    }
}

void smoe_metal_register_buffer(SmoeMetalCtx* ctx, const void* ptr, size_t size_in_bytes) {
    if (!ctx || !ptr) return;

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
        MTLSize gridGroups = MTLSizeMake(rows[i], 1, 1);
        [enc dispatchThreadgroups:gridGroups threadsPerThreadgroup:tgSize];
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
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
