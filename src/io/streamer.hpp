// ═══════════════════════════════════════════════════════════════
// streamer.hpp — S-MoE Engine · High-Velocity Direct I/O Streamer
// ═══════════════════════════════════════════════════════════════
// Phase 2 — Week 2
//
// Design invariants (MUST NEVER be violated):
//   ① F_NOCACHE on every fd — kernel page cache fully bypassed.
//   ② All slot data buffers are posix_memalign(16 KB)-aligned.
//   ③ Expert blobs are read via pread() — concurrent, offset-based,
//      no seek, no lock.
//   ④ Ring slot transitions are CAS-only: EMPTY→LOADING→READY→CONSUMED.
//   ⑤ Zero heap allocations inside prefetch() / claim_ready() / release().
//   ⑥ Worker threads never call malloc, new, or any blocking mutex.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include "../common.hpp"

#include <atomic>
#include <cstdint>

namespace smoe::io {

// ─────────────────────────────────────────────────────────────
// Ring slot state machine — all transitions are atomic CAS.
//
//   EMPTY ──► LOADING ──► READY ──► CONSUMED ──► EMPTY
//     ▲                                              │
//     └──────────────────────────────────────────────┘
// ─────────────────────────────────────────────────────────────
enum class SlotState : uint32_t {
    EMPTY    = 0,   // free; available for a new prefetch request
    LOADING  = 1,   // owned by an I/O worker; pread() in progress
    READY    = 2,   // data is in UMA; GPU / Metal bridge may consume
    CONSUMED = 3,   // GPU done; pending reclaim back to EMPTY
};

// ─────────────────────────────────────────────────────────────
// RingSlot — one expert-block container.
//
// alignas(64): each slot occupies exactly one L1 cache line.
// This prevents false sharing when multiple worker threads scan
// the ring concurrently.
//
// Layout (64 bytes):
//   [state: 4][layer_id: 4][expert_id: 4][_pad: 4]  = 16
//   [data*: 8][data_size: 8]                          = 16
//   [_pad2: 32]                                        = 32
//                                           total = 64 ✓
// ─────────────────────────────────────────────────────────────
struct alignas(64) RingSlot {
    std::atomic<SlotState> state     { SlotState::EMPTY };
    uint32_t               layer_id  { 0xFFFFFFFF };
    uint32_t               expert_id { 0xFFFFFFFF };
    std::atomic<uint32_t>  ref_count { 0 };
    std::atomic<uint64_t>  last_used_tick { 0 };
    uint8_t*               data      { nullptr };  // posix_memalign 16 KB-aligned
    uint64_t               data_size { 0 };        // bytes actually loaded by pread()
    uint8_t                _pad1[24] {};
};
static_assert(sizeof(RingSlot) == 64,
    "RingSlot must be exactly one cache line (64 bytes).");

// ─────────────────────────────────────────────────────────────
// Streamer — the river.
//
// Manages a fixed-size ring of pre-allocated, page-aligned UMA
// buffers and a pool of background worker threads that fill
// them via Direct I/O pread() calls against the .smoe vault.
//
// Thread model:
//   Caller thread   →  prefetch() / claim_ready() / release()
//   N worker threads →  pop request queue, pread(), mark READY
//
// All synchronisation: std::atomic only. No OS mutex, ever.
// ─────────────────────────────────────────────────────────────
class Streamer {
public:
    // Construct the streamer and launch worker threads.
    //
    //   smoe_path      — path to the .smoe vault
    //   ring_size      — number of ring slots (≥ K × max_active_experts)
    //   num_workers    — background I/O thread count (≤ NVMe queue depth)
    //   slot_capacity  — bytes per slot (≥ largest expert padded_size)
    explicit Streamer(const char* smoe_path,
                      uint32_t    ring_size     = 48,
                      uint32_t    num_workers   = 4,
                      uint64_t    slot_capacity = 8ULL * 1024 * 1024);  // 8 MB

    // Drain in-flight reads, join all workers, free all memory.
    ~Streamer();

    // Enqueue an asynchronous prefetch for (layer_id, expert_id).
    // Idempotent: returns true immediately if the expert is already
    // LOADING or READY in the ring (no duplicate load).
    // Returns false under back-pressure (ring saturated or queue full).
    // Non-blocking. Zero allocations.
    [[nodiscard]] bool prefetch(uint32_t layer_id, uint32_t expert_id) noexcept;

    // Claim the next READY slot. Caller takes ownership; must call
    // claim_ready() / release() when the Metal kernel has finished with the data.
    // Returns nullptr if no slot is READY (non-blocking).
    // Zero allocations.
    [[nodiscard]] RingSlot* claim_ready() noexcept;

    // Claim a specific READY slot for a given layer and expert.
    // Caller takes ownership; must call release() when done.
    // Returns nullptr if the specific expert is not READY (non-blocking).
    [[nodiscard]] RingSlot* claim_specific(uint32_t layer_id, uint32_t expert_id) noexcept;

    // Debugging helpers
    void debug_print() const;
    uint32_t debug_slot_state(uint32_t layer_id, uint32_t expert_id) const noexcept;
    void debug_dump_ring() const noexcept;

    // Release a slot the GPU is done with.
    // Atomically transitions CONSUMED → EMPTY.
    // Zero allocations.
    void release(RingSlot* slot) noexcept;

    // Reclaim/garbage collect any READY slots whose (layer, expert) is NOT active.
    // Zero allocations.
    void prune_slots(bool (*is_active)(uint32_t, uint32_t, void*), void* ctx) noexcept;

    // Gracefully stop all worker threads and join them.
    void shutdown() noexcept;

    // ── Telemetry counters (lock-free reads for live display) ──
    [[nodiscard]] uint64_t bytes_read()   const noexcept;
    [[nodiscard]] uint64_t hit_count()    const noexcept;
    [[nodiscard]] uint64_t miss_count()   const noexcept;
    [[nodiscard]] uint32_t ring_size()    const noexcept;
    [[nodiscard]] uint32_t ready_count()  const noexcept;
    void print_debug_states()             const noexcept;
    
    // ── Memory Pool access (for Metal registration) ──
    [[nodiscard]] const void* pool_data() const noexcept;
    [[nodiscard]] uint64_t    pool_size() const noexcept;
    
    [[nodiscard]] const void* get_slot_ptr(uint32_t index) const noexcept;
    [[nodiscard]] uint64_t    get_slot_bytes() const noexcept;

    Streamer(const Streamer&)            = delete;
    Streamer& operator=(const Streamer&) = delete;
    Streamer(Streamer&&)                 = delete;
    Streamer& operator=(Streamer&&)      = delete;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace smoe::io
