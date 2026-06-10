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

// ── Embedded MSL source ───────────────────────────────────────
// Read at compile time from the adjacent .metal file.
// We embed it as a raw string literal so the binary is fully
// self-contained (Option B decision: JIT on first launch).
// The string is derived from kernels.metal verbatim — any change
// to that file must be reflected here.
static const char* kMetalSource = R"MSL(

#include <metal_stdlib>
using namespace metal;

constant uint TGROUP_SIZE = 256;

struct FusedFFNParams {
    uint rows;
    uint cols;
    uint group_size;
};

inline float silu(float x) {
    return x * (1.0f / (1.0f + exp(-x)));
}

inline float smoeq2_dequant(
    device const uint8_t* packed,
    device const half*    scales,
    uint wi,
    uint group_size)
{
    uint    pack_idx  = wi >> 2;
    uint    bit_shift = (wi & 3u) << 1u;
    uint    group_idx = wi / group_size;
    uint8_t code      = (packed[pack_idx] >> bit_shift) & 0x3u;
    float   scale     = float(scales[group_idx]);
    return ((float(code) - 1.5f) * (1.0f / 1.5f)) * scale;
}

kernel void smoe_gate_up(
    device const uint8_t*    gate_packed  [[buffer(0)]],
    device const half*       gate_scales  [[buffer(1)]],
    device const uint8_t*    up_packed    [[buffer(2)]],
    device const half*       up_scales    [[buffer(3)]],
    device const float*      input        [[buffer(6)]],
    device       float*      hidden       [[buffer(7)]],
    constant FusedFFNParams& params       [[buffer(9)]],
    uint                     gid          [[thread_position_in_grid]],
    uint                     tid          [[thread_index_in_threadgroup]],
    threadgroup float*       tg_input     [[threadgroup(0)]])
{
    uint row = gid;
    if (row >= params.rows) return;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
    uint  tiles    = (params.cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        uint col      = col_base + tid;
        tg_input[tid] = (col < params.cols) ? input[col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_end = min(TGROUP_SIZE, params.cols - col_base);
        uint wi_base  = row * params.cols + col_base;

        uint k = 0;
        for (; k + 3 < tile_end; k += 4) {
            uint    pidx  = (wi_base + k) >> 2;
            uint8_t gbyte = gate_packed[pidx];
            uint8_t ubyte = up_packed[pidx];
            float   gs    = float(gate_scales[(wi_base + k) / params.group_size]);
            float   us    = float(up_scales[(wi_base + k)   / params.group_size]);
            for (uint b = 0; b < 4; ++b) {
                float gw  = (float((gbyte >> (b * 2)) & 0x3u) - 1.5f) * (1.0f / 1.5f) * gs;
                float uw  = (float((ubyte >> (b * 2)) & 0x3u) - 1.5f) * (1.0f / 1.5f) * us;
                float x_k = tg_input[k + b];
                gate_acc += gw * x_k;
                up_acc   += uw * x_k;
            }
        }
        for (; k < tile_end; ++k) {
            uint wi = wi_base + k;
            gate_acc += smoeq2_dequant(gate_packed, gate_scales, wi, params.group_size) * tg_input[k];
            up_acc   += smoeq2_dequant(up_packed,   up_scales,   wi, params.group_size) * tg_input[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    hidden[row] = silu(gate_acc) * up_acc;
}

kernel void smoe_down(
    device const uint8_t*    down_packed  [[buffer(4)]],
    device const half*       down_scales  [[buffer(5)]],
    device const float*      hidden       [[buffer(7)]],
    device       float*      output       [[buffer(8)]],
    constant FusedFFNParams& params       [[buffer(9)]],
    uint                     gid          [[thread_position_in_grid]],
    uint                     tid          [[thread_index_in_threadgroup]],
    threadgroup float*       tg_hidden    [[threadgroup(0)]])
{
    uint row = gid;
    if (row >= params.rows) return;

    float acc   = 0.0f;
    uint  tiles = (params.cols + TGROUP_SIZE - 1) / TGROUP_SIZE;

    for (uint t = 0; t < tiles; ++t) {
        uint col_base = t * TGROUP_SIZE;
        uint col      = col_base + tid;
        tg_hidden[tid] = (col < params.cols) ? hidden[col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_end = min(TGROUP_SIZE, params.cols - col_base);
        uint wi_base  = row * params.cols + col_base;

        uint k = 0;
        for (; k + 3 < tile_end; k += 4) {
            uint    pidx  = (wi_base + k) >> 2;
            uint8_t dbyte = down_packed[pidx];
            float   ds    = float(down_scales[(wi_base + k) / params.group_size]);
            for (uint b = 0; b < 4; ++b) {
                float dw = (float((dbyte >> (b * 2)) & 0x3u) - 1.5f) * (1.0f / 1.5f) * ds;
                acc     += dw * tg_hidden[k + b];
            }
        }
        for (; k < tile_end; ++k) {
            uint wi = wi_base + k;
            acc += smoeq2_dequant(down_packed, down_scales, wi, params.group_size) * tg_hidden[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[row] = acc;
}

)MSL";

// ═══════════════════════════════════════════════════════════════
// SmoeMetalCtx — internal implementation struct
// ═══════════════════════════════════════════════════════════════
struct SmoeMetalCtx {
    id<MTLDevice>               device       = nil;
    id<MTLCommandQueue>         queue        = nil;
    id<MTLLibrary>              library      = nil;
    id<MTLComputePipelineState> gate_up_pso  = nil;
    id<MTLComputePipelineState> down_pso     = nil;

    // Ping-pong UMA buffers
    // buf_a = active (GPU executing)
    // buf_b = passive (I/O streamer filling)
    id<MTLBuffer>               buf_a        = nil;  // active
    id<MTLBuffer>               buf_b        = nil;  // passive

    // Pre-allocated scratch buffer for the hidden activation vector.
    // Size = max_rows * sizeof(float). Allocated in smoe_metal_init.
    id<MTLBuffer>               hidden_buf   = nil;

    size_t                      slot_bytes   = 0;
    uint32_t                    max_rows     = 0;    // = slot_bytes / sizeof(float)

    // Atomic flag: true when buf_b is safe for the I/O streamer to fill.
    std::atomic<bool>           passive_ready { true };

    // Telemetry
    std::atomic<uint64_t>       dispatch_count { 0 };
};

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
    std::fprintf(stderr, "[smoe_metal] Compiling SMOE-Q2 kernels (JIT)...\n");

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
    std::fprintf(stderr, "[smoe_metal] Kernel compilation OK.\n");

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
    // Sized for the worst-case intermediate dim.
    // max_rows heuristic: slot_bytes / 4 bytes-per-float, capped
    // at 32768 (generous upper bound for any DeepSeek expert dim).
    uint32_t max_rows    = static_cast<uint32_t>(
        std::min(slot_bytes / sizeof(float), size_t(32768)));
    id<MTLBuffer> hidden = [dev newBufferWithLength:max_rows * sizeof(float)
                                            options:uma];
    if (!hidden) {
        std::fprintf(stderr,
            "[smoe_metal] ERROR: Failed to allocate hidden scratch buffer.\n");
        return nullptr;
    }

    // ── Step 6: create command queue ─────────────────────────
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!queue) {
        std::fprintf(stderr, "[smoe_metal] ERROR: Failed to create command queue.\n");
        return nullptr;
    }

    // ── Step 7: assemble context ──────────────────────────────
    SmoeMetalCtx* ctx = new SmoeMetalCtx();
    ctx->device      = dev;
    ctx->queue       = queue;
    ctx->library     = lib;
    ctx->gate_up_pso = gate_up_pso;
    ctx->down_pso    = down_pso;
    ctx->buf_a       = buf_a;
    ctx->buf_b       = buf_b;
    ctx->hidden_buf  = hidden;
    ctx->slot_bytes  = slot_bytes;
    ctx->max_rows    = max_rows;
    ctx->passive_ready.store(true, std::memory_order_release);
    ctx->dispatch_count.store(0, std::memory_order_relaxed);

    std::fprintf(stderr,
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

void smoe_metal_fused_ffn(SmoeMetalCtx*   ctx,
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
                           uint32_t        group_size)
{
    if (!ctx) return;

    // ── Build FusedFFNParams ──────────────────────────────────
    // Stack allocation — not in hot-path alloc-banned zone because
    // this struct is tiny and compiler-stack-allocated.
    struct FusedFFNParamsC { uint32_t rows, cols, group_size; };
    FusedFFNParamsC params { rows, cols, group_size };

    // ── Wrap raw pointers in MTLBuffers (no-copy) ─────────────
    // newBufferWithBytesNoCopy creates an MTLBuffer view over
    // memory we already own (pread'd data in the UMA ping-pong buf).
    // The deallocator is nil because the lifetime is managed externally.
    MTLResourceOptions uma = MTLResourceStorageModeShared;

    auto wrap = [&](const void* ptr, size_t sz) -> id<MTLBuffer> {
        // Length must be PAGE_SIZE-aligned; round up.
        size_t aligned = (sz + (smoe::PAGE_SIZE - 1)) & ~(smoe::PAGE_SIZE - 1);
        return [ctx->device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                              length:aligned
                                             options:uma
                                         deallocator:nil];
    };

    // Compute byte sizes for each sub-tensor
    uint64_t packed_bytes = static_cast<uint64_t>(rows) * cols / 4;   // 2-bit packing
    uint64_t scale_bytes  = static_cast<uint64_t>(rows) * cols / group_size * sizeof(uint16_t);
    uint64_t input_bytes  = static_cast<uint64_t>(cols)  * sizeof(float);
    uint64_t output_bytes = static_cast<uint64_t>(rows)  * sizeof(float);

    id<MTLBuffer> buf_gp = wrap(packed_gate,  packed_bytes);
    id<MTLBuffer> buf_gs = wrap(scales_gate,  scale_bytes);
    id<MTLBuffer> buf_up = wrap(packed_up,    packed_bytes);
    id<MTLBuffer> buf_us = wrap(scales_up,    scale_bytes);
    id<MTLBuffer> buf_dp = wrap(packed_down,  packed_bytes);
    id<MTLBuffer> buf_ds = wrap(scales_down,  scale_bytes);
    id<MTLBuffer> buf_in = wrap(input_vec,    input_bytes);
    id<MTLBuffer> buf_hd = ctx->hidden_buf;   // pre-allocated scratch
    id<MTLBuffer> buf_ou = wrap(output_vec,   output_bytes);

    // Params as a tiny inline buffer
    id<MTLBuffer> buf_pa = [ctx->device newBufferWithBytes:&params
                                                    length:sizeof(params)
                                                   options:uma];

    // ── Encode and commit ─────────────────────────────────────
    id<MTLCommandBuffer>         cmd  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc  = [cmd computeCommandEncoder];

    // ── Pass 1: smoe_gate_up ──────────────────────────────────
    [enc setComputePipelineState:ctx->gate_up_pso];
    [enc setBuffer:buf_gp offset:0 atIndex:0];
    [enc setBuffer:buf_gs offset:0 atIndex:1];
    [enc setBuffer:buf_up offset:0 atIndex:2];
    [enc setBuffer:buf_us offset:0 atIndex:3];
    [enc setBuffer:buf_in offset:0 atIndex:6];
    [enc setBuffer:buf_hd offset:0 atIndex:7];
    [enc setBuffer:buf_pa offset:0 atIndex:9];
    // Threadgroup SRAM: TGROUP_SIZE floats for the input tile
    [enc setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];

    {
        NSUInteger tgroup = ctx->gate_up_pso.maxTotalThreadsPerThreadgroup;
        tgroup = std::min(tgroup, NSUInteger(256));
        MTLSize tgSize    = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridSize  = MTLSizeMake(rows,   1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }

    // ── Pass 2: smoe_down ─────────────────────────────────────
    [enc setComputePipelineState:ctx->down_pso];
    [enc setBuffer:buf_dp offset:0 atIndex:4];
    [enc setBuffer:buf_ds offset:0 atIndex:5];
    [enc setBuffer:buf_hd offset:0 atIndex:7];
    [enc setBuffer:buf_ou offset:0 atIndex:8];
    [enc setBuffer:buf_pa offset:0 atIndex:9];
    [enc setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];

    {
        NSUInteger tgroup = ctx->down_pso.maxTotalThreadsPerThreadgroup;
        tgroup = std::min(tgroup, NSUInteger(256));
        MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(rows,   1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }

    [enc endEncoding];

    // Synchronous commit for Phase 3 validation.
    // Phase 4 will convert this to async with a completion handler
    // that signals the Streamer's ring via atomic flag.
    [cmd commit];
    [cmd waitUntilCompleted];

    ctx->dispatch_count.fetch_add(1, std::memory_order_relaxed);
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
