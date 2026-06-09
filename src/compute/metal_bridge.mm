// ═══════════════════════════════════════════════════════════════
// metal_bridge.mm — S-MoE Engine · Objective-C++ Metal Bridge
// ═══════════════════════════════════════════════════════════════
// Phase 3 — Week 3 implementation target.
//
// Will implement:
//   • MTLDevice / MTLCommandQueue initialisation.
//   • Ping-pong MTLBuffer allocation in Unified Memory.
//   • Zero-copy pointer injection (newBufferWithBytesNoCopy).
//   • Dispatch of the SMOE-Q2 dequantising FFN Metal kernel.
//   • Atomic buffer-swap signalling to the I/O streamer.
// ═══════════════════════════════════════════════════════════════

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_bridge.h"
#include "../common.hpp"

// TODO — Week 3

struct SmoeMetalCtx {
    id<MTLDevice>       device       = nil;
    id<MTLCommandQueue> queue        = nil;
    id<MTLLibrary>      library      = nil;
    id<MTLComputePipelineState> fused_ffn_pso = nil;
    // Ping-pong buffers (A = active GPU, B = filling via I/O)
    id<MTLBuffer>       buf_a        = nil;
    id<MTLBuffer>       buf_b        = nil;
    size_t              slot_bytes   = 0;
};

SmoeMetalCtx* smoe_metal_init(size_t slot_bytes) {
    (void)slot_bytes;
    // TODO Week 3
    return nullptr;
}

void smoe_metal_fused_ffn(SmoeMetalCtx* ctx,
                           const uint8_t*  packed_data,
                           const uint16_t* scales_data,
                           const float*    input_vec,
                           float*          output_vec,
                           uint32_t rows,
                           uint32_t cols,
                           uint32_t group_size)
{
    (void)ctx; (void)packed_data; (void)scales_data;
    (void)input_vec; (void)output_vec;
    (void)rows; (void)cols; (void)group_size;
    // TODO Week 3
}

void smoe_metal_swap_buffers(SmoeMetalCtx* ctx) {
    if (!ctx) return;
    // TODO Week 3 — atomic ping-pong swap
}

void smoe_metal_destroy(SmoeMetalCtx* ctx) {
    delete ctx;
}
