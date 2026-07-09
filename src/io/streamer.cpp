// ═══════════════════════════════════════════════════════════════
// streamer.cpp — S-MoE Engine · High-Velocity Direct I/O Streamer
// ═══════════════════════════════════════════════════════════════
// Phase 2 — Week 2
//
// Strategy:
//   • Open the .smoe vault once with F_NOCACHE — zero kernel cache.
//   • Parse SmoeHeader + ExpertTable at construction → O(1) lookup map.
//   • Pre-allocate a contiguous pool of 16 KB-aligned UMA buffers
//     (one per ring slot) via posix_memalign — zero runtime allocation.
//   • N worker threads each pop PrefetchRequests from a lock-free
//     MPMC queue (Dmitry Vyukov's algorithm), claim an EMPTY ring
//     slot via CAS, pread() the expert blob in, mark READY.
//   • Caller (Metal bridge) claims READY slots, uses them, releases.
//   • All synchronisation: std::atomic. No mutex. No condition_variable.
// ═══════════════════════════════════════════════════════════════

#include "streamer.hpp"
#include "../common.hpp"

// ── POSIX / macOS system headers ─────────────────────────────
#include <fcntl.h>      // open(), fcntl(), F_NOCACHE
#include <sys/types.h>
#include <unistd.h>     // pread(), close()

// ── C++ standard library ─────────────────────────────────────
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdlib>      // posix_memalign, free
#include <cstring>      // memset
#include <new>          // std::hardware_destructive_interference_size
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace smoe::io {

// ─────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────

struct PrefetchRequest {
    uint32_t  layer_id;
    uint32_t  expert_id;
    RingSlot* slot;
};

[[nodiscard]] static constexpr uint64_t expert_key(
    uint32_t layer_id, uint32_t expert_id) noexcept
{
    return (static_cast<uint64_t>(layer_id) << 32) | expert_id;
}

// ─────────────────────────────────────────────────────────────
// Lock-free MPMC bounded queue — Dmitry Vyukov's algorithm.
//
// Multiple producers (main loop enqueuing prefetch requests) and
// multiple consumers (worker threads dequeuing them).
//
// Capacity must be a power of two.
// push() returns false when full  (back-pressure signal).
// pop()  returns false when empty (worker spins / yields).
//
// No heap allocation after construction. No mutex.
// ─────────────────────────────────────────────────────────────
template<size_t CAPACITY>
class MPMCQueue {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0,
        "CAPACITY must be a power of two");

    struct alignas(64) Cell {
        std::atomic<size_t> sequence { 0 };
        PrefetchRequest     data     {};
    };

    alignas(64) Cell                buffer_[CAPACITY];
    alignas(64) std::atomic<size_t> enqueue_pos_ { 0 };
    alignas(64) std::atomic<size_t> dequeue_pos_ { 0 };

public:
    MPMCQueue() noexcept {
        for (size_t i = 0; i < CAPACITY; ++i)
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }

    [[nodiscard]] bool push(const PrefetchRequest& req) noexcept {
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Cell*  cell = &buffer_[pos & (CAPACITY - 1)];
            size_t seq  = cell->sequence.load(std::memory_order_acquire);
            auto   diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    cell->data = req;
                    cell->sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;   // queue full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] bool pop(PrefetchRequest& req) noexcept {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Cell*  cell = &buffer_[pos & (CAPACITY - 1)];
            size_t seq  = cell->sequence.load(std::memory_order_acquire);
            auto   diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    req = cell->data;
                    cell->sequence.store(pos + CAPACITY, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;   // queue empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    size_t get_enqueue_pos() const noexcept { return enqueue_pos_.load(std::memory_order_relaxed); }
    size_t get_dequeue_pos() const noexcept { return dequeue_pos_.load(std::memory_order_relaxed); }
};

static constexpr size_t REQUEST_QUEUE_CAPACITY = 1024;  // power of two

// ─────────────────────────────────────────────────────────────
// Streamer::Impl — all private state
// ─────────────────────────────────────────────────────────────
struct Streamer::Impl {
    // ── Vault file ────────────────────────────────────────────
    int fd { -1 };  // main thread fd (used for parsing)
    std::string vault_path; // used by workers to open their own fds

    // ── Expert index (built once at startup) ──────────────────
    // key: expert_key(layer_id, expert_id) → ExpertEntry
    std::unordered_map<uint64_t, smoe::ExpertEntry> expert_index;

    // ── Ring buffer ───────────────────────────────────────────
    uint32_t  ring_sz   { 0 };
    uint64_t  slot_cap  { 0 };        // bytes per slot
    RingSlot* slots     { nullptr };  // ring_sz slots, cache-line-aligned
    uint8_t*  data_pool { nullptr };  // one posix_memalign block, 16 KB-aligned

    // ── Request queues ────────────────────────────────────────
    // Two priorities: workers drain request_queue (demand) first and
    // touch spec_queue (speculative next-token bets) only when demand
    // is empty. Same lock-free MPMC structure for both.
    MPMCQueue<REQUEST_QUEUE_CAPACITY> request_queue;
    MPMCQueue<REQUEST_QUEUE_CAPACITY> spec_queue;

    // ── LRU Cache Tick ────────────────────────────────────────
    std::atomic<uint64_t> global_tick { 0 };

    // ── Worker pool ───────────────────────────────────────────
    uint32_t                 num_workers { 0 };
    std::atomic<bool>        shutdown    { false };
    std::vector<std::thread> workers;

    // ── Telemetry (relaxed atomics — approximate is fine) ─────
    std::atomic<uint64_t> stat_bytes_read { 0 };
    std::atomic<uint64_t> stat_hits       { 0 };
    std::atomic<uint64_t> stat_misses     { 0 };

    // ── Worker thread entry point (static to avoid naming Impl from outside) ──
    static void run(Impl* im, uint32_t worker_id);
};

// ─────────────────────────────────────────────────────────────
// Vault parser — reads SmoeHeader + ExpertTable at startup.
// Called once; regular (non-Direct-I/O) reads are fine here.
// ─────────────────────────────────────────────────────────────
static void parse_vault(int fd,
    std::unordered_map<uint64_t, smoe::ExpertEntry>& index)
{
    // ── Header ────────────────────────────────────────────────
    smoe::SmoeHeader hdr {};
    ssize_t n = ::pread(fd, &hdr, sizeof(hdr), 0);
    if (n != static_cast<ssize_t>(sizeof(hdr)))
        throw std::runtime_error("smoe: failed to read file header");

    if (!smoe::magic_valid(hdr.magic))
        throw std::runtime_error("smoe: invalid magic bytes — not a .smoe vault");

    if (hdr.version != smoe::SMOE_VERSION)
        throw std::runtime_error(
            "smoe: unsupported format version " + std::to_string(hdr.version));

    if (hdr.total_experts == 0)
        throw std::runtime_error("smoe: vault contains zero experts");

    // ── Expert table ──────────────────────────────────────────
    const size_t table_bytes =
        static_cast<size_t>(hdr.total_experts) * sizeof(smoe::ExpertEntry);

    std::vector<smoe::ExpertEntry> table(hdr.total_experts);
    n = ::pread(fd, table.data(), table_bytes,
                static_cast<off_t>(hdr.table_offset));

    if (static_cast<size_t>(n) != table_bytes)
        throw std::runtime_error("smoe: failed to read expert table");

    // ── Build O(1) lookup map ─────────────────────────────────
    index.reserve(hdr.total_experts);
    for (const auto& e : table)
        index[expert_key(e.layer_id, e.expert_id)] = e;
}

// ─────────────────────────────────────────────────────────────
// Worker thread body.
//
// Each worker:
//   1. Pops a PrefetchRequest from the MPMC queue (spins with
//      exponential back-off when idle to avoid burning CPU).
//   2. Claims an EMPTY ring slot via CAS (scans cyclically from
//      worker_id offset to spread load and reduce contention).
//   3. Issues a Direct I/O pread() into the slot's 16 KB-aligned
//      data buffer.
//   4. Stores metadata and atomically marks the slot READY.
// ─────────────────────────────────────────────────────────────
void Streamer::Impl::run(Impl* impl, uint32_t worker_id)
{
    PrefetchRequest req {};
    uint32_t backoff = 1;              // exponential yield backoff

    // macOS kernel serializes F_NOCACHE preads on the same file descriptor.
    // Give every thread its own independent FD to unlock full NVMe queue depth.
    int local_fd = ::open(impl->vault_path.c_str(), O_RDONLY);
    if (local_fd >= 0) {
        ::fcntl(local_fd, F_NOCACHE, 1);
    }

    while (!impl->shutdown.load(std::memory_order_relaxed)) {

        // ── 1. Get a pending request — demand first, then spec ─
        if (!impl->request_queue.pop(req) && !impl->spec_queue.pop(req)) {
            // No work — exponential yield to avoid busy-spin
            for (uint32_t i = 0; i < backoff; ++i)
                std::this_thread::yield();
            backoff = std::min(backoff * 2u, 256u);
            continue;
        }
        backoff = 1;  // reset: work was available

        // ── 2. Resolve expert file offset ─────────────────────
        auto it = impl->expert_index.find(expert_key(req.layer_id, req.expert_id));
        if (it == impl->expert_index.end()) {
            // Unknown expert — should never happen with a valid vault
            std::fprintf(stderr, "[worker] ERROR: Unknown expert requested: layer=%u, expert=%u\n", req.layer_id, req.expert_id);
            impl->stat_misses.fetch_add(1, std::memory_order_relaxed);
            if (req.slot) {
                req.slot->state.store(SlotState::EMPTY, std::memory_order_release);
            }
            continue;
        }
        const smoe::ExpertEntry& entry = it->second;

        // ── 3. We ALREADY claimed the ring slot in prefetch() ─
        RingSlot* slot = req.slot;
        if (!slot) continue;

        // ── 4. Direct I/O read into 16 KB-aligned slot buffer ─
        //
        //  entry.byte_offset:  guaranteed 16 KB-aligned by shatter_moe.py
        //  slot->data:         guaranteed 16 KB-aligned by posix_memalign
        //  F_NOCACHE already set on fd — NVMe DMA straight into UMA.
        const ssize_t n = ::pread(
            local_fd >= 0 ? local_fd : impl->fd,
            slot->data,
            static_cast<size_t>(entry.padded_size),
            static_cast<off_t>(entry.byte_offset)
        );

        if (n < 0 ||
            static_cast<uint64_t>(n) != entry.padded_size)
        {
            if (n < 0) {
                std::perror("[worker] pread failed");
            } else {
                std::fprintf(stderr, "[worker] pread short read: %zd != %llu\n", n, (unsigned long long)entry.padded_size);
            }
            // I/O error — release slot and count miss
            slot->state.store(SlotState::EMPTY, std::memory_order_release);
            impl->stat_misses.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // ── 5. Fill metadata and flip to READY ────────────────
        // Writes to non-atomic fields are safe because we own
        // this slot exclusively (state == LOADING).
        slot->data_size  = static_cast<uint64_t>(n);

        // Release fence: all prior writes must be visible to any thread
        // that subsequently observes state == READY.
        slot->state.store(SlotState::READY, std::memory_order_release);

        impl->stat_bytes_read.fetch_add(static_cast<uint64_t>(n),
                                        std::memory_order_relaxed);
        impl->stat_hits.fetch_add(1, std::memory_order_relaxed);
    }
}

// ─────────────────────────────────────────────────────────────
// Streamer — public API implementation
// ─────────────────────────────────────────────────────────────

Streamer::Streamer(const char* smoe_path,
                   uint32_t    ring_size,
                   uint32_t    num_workers,
                   uint64_t    slot_capacity)
    : impl_(new Impl)
{
    Impl& im = *impl_;
    im.ring_sz      = ring_size;
    im.slot_cap     = slot_capacity;
    im.num_workers  = num_workers;
    im.vault_path   = smoe_path;

    // ── Open vault file with Direct I/O ──────────────────────
    im.fd = ::open(smoe_path, O_RDONLY);
    if (im.fd < 0)
        throw std::runtime_error(
            std::string("smoe: cannot open vault: ") + smoe_path);

    // Disable kernel unified page cache — every byte goes straight
    // from the NVMe controller into our pre-allocated UMA buffers.
    if (::fcntl(im.fd, F_NOCACHE, 1) < 0)
        throw std::runtime_error("smoe: fcntl(F_NOCACHE) failed");

    // ── Parse SmoeHeader + ExpertTable ───────────────────────
    parse_vault(im.fd, im.expert_index);

    // ── Pre-allocate ring slots (cache-line-aligned array) ────
    // Use aligned_alloc so that each RingSlot starts on a 64-byte
    // boundary, preserving the alignas(64) guarantee.
    const size_t slots_bytes = ring_size * sizeof(RingSlot);
    void* slots_raw = nullptr;
    if (::posix_memalign(&slots_raw, 64, slots_bytes) != 0)
        throw std::runtime_error("smoe: posix_memalign failed for ring slots");
    im.slots = new (slots_raw) RingSlot[ring_size];  // placement-new to init atomics

    // ── Pre-allocate data pool (16 KB-aligned contiguous block) ─
    // All slot data pointers are carved out of this one block.
    // posix_memalign guarantees 16 KB alignment for the whole pool,
    // and each slot's offset is a multiple of slot_capacity, so each
    // slot pointer is also 16 KB-aligned (provided slot_capacity is
    // a multiple of PAGE_SIZE, which callers must ensure).
    assert(slot_capacity % smoe::PAGE_SIZE == 0 &&
           "slot_capacity must be a multiple of PAGE_SIZE (16 KB)");

    const size_t pool_bytes = static_cast<size_t>(ring_size) * slot_capacity;
    void* pool_raw = nullptr;
    if (::posix_memalign(&pool_raw, smoe::PAGE_SIZE, pool_bytes) != 0)
        throw std::runtime_error("smoe: posix_memalign failed for data pool");

    im.data_pool = static_cast<uint8_t*>(pool_raw);
    std::memset(im.data_pool, 0, pool_bytes);

    // Wire each slot to its region in the data pool
    for (uint32_t i = 0; i < ring_size; ++i)
        im.slots[i].data = im.data_pool + static_cast<size_t>(i) * slot_capacity;

    // ── Launch worker threads ─────────────────────────────────
    im.workers.reserve(num_workers);
    for (uint32_t w = 0; w < num_workers; ++w)
        im.workers.emplace_back(Impl::run, &im, w);
}

Streamer::~Streamer() {
    shutdown();
    delete impl_;
}

void Streamer::shutdown() noexcept {
    if (!impl_) return;
    Impl& im = *impl_;

    // Signal all workers to stop after their current pread() completes
    im.shutdown.store(true, std::memory_order_release);

    for (auto& t : im.workers)
        if (t.joinable()) t.join();
    im.workers.clear();

    // Release ring and pool memory
    if (im.slots) {
        for (uint32_t i = 0; i < im.ring_sz; ++i)
            im.slots[i].~RingSlot();  // explicit destructor for placement-new'd objects
        std::free(im.slots);
        im.slots = nullptr;
    }
    if (im.data_pool) {
        std::free(im.data_pool);
        im.data_pool = nullptr;
    }
    if (im.fd >= 0) {
        ::close(im.fd);
        im.fd = -1;
    }
}

bool Streamer::prefetch(uint32_t layer_id, uint32_t expert_id,
                        bool speculative) noexcept {
    Impl& im = *impl_;

    // ── Idempotency check: already in the ring? ───────────────
    // Scan for any slot that is LOADING or READY for this expert.
    // O(ring_sz) — acceptable; called at most K×experts_per_step per token.
    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        const SlotState s = im.slots[i].state.load(std::memory_order_relaxed);
        if (s != SlotState::EMPTY &&
            im.slots[i].layer_id  == layer_id &&
            im.slots[i].expert_id == expert_id)
        {
            im.slots[i].last_used_tick.store(im.global_tick.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
            im.slots[i].last_used_tick.store(im.global_tick.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
            return true;  // already queued or ready — no duplicate load
        }
    }

    // ── Claim an EMPTY ring slot via CAS ───────────────
    RingSlot* slot = nullptr;
    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        SlotState expected = SlotState::EMPTY;
        if (im.slots[i].state.compare_exchange_strong(
                expected, SlotState::LOADING,
                std::memory_order_acquire,
                std::memory_order_relaxed))
        {
            slot = &im.slots[i];
            break;
        }
    }

    // ── Evict the least-recently-used idle slot if none EMPTY ──
    // Considers CONSUMED and READY slots alike, but ONLY with
    // ref_count == 0: a CONSUMED slot with a live reference is still
    // being read by the GPU — evicting it (as the old first-fit
    // CONSUMED scan did) lets the worker pread() over data mid-kernel.
    // LRU order matters now that the ring retains slots across tokens
    // as a cache instead of being mass-pruned every step.
    if (!slot) {
        uint64_t oldest_tick = UINT64_MAX;
        int best_idx = -1;
        for (uint32_t i = 0; i < im.ring_sz; ++i) {
            SlotState s = im.slots[i].state.load(std::memory_order_relaxed);
            if ((s == SlotState::CONSUMED || s == SlotState::READY) &&
                im.slots[i].ref_count.load(std::memory_order_relaxed) == 0)
            {
                uint64_t tick = im.slots[i].last_used_tick.load(std::memory_order_relaxed);
                if (tick < oldest_tick) {
                    oldest_tick = tick;
                    best_idx = static_cast<int>(i);
                }
            }
        }
        if (best_idx >= 0) {
            SlotState expected =
                im.slots[best_idx].state.load(std::memory_order_relaxed);
            if ((expected == SlotState::CONSUMED || expected == SlotState::READY) &&
                im.slots[best_idx].state.compare_exchange_strong(
                    expected, SlotState::LOADING,
                    std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                slot = &im.slots[best_idx];
            }
        }
    }

    if (!slot) {
        return false;
    }

    // Set metadata NOW so idempotency check catches subsequent prefetch calls
    slot->layer_id = layer_id;
    slot->expert_id = expert_id;
    slot->ref_count.store(0, std::memory_order_relaxed);
    // Fresh tick: without this a newly loaded slot keeps its previous
    // occupant's (old) tick and becomes the LRU eviction victim before
    // it is ever read — a large prefetch burst then evicts its own head.
    slot->last_used_tick.store(
        im.global_tick.fetch_add(1, std::memory_order_relaxed),
        std::memory_order_relaxed);

    // ── Enqueue the request (priority by class) ───────────────
    const bool pushed = speculative
        ? im.spec_queue.push({ layer_id, expert_id, slot })
        : im.request_queue.push({ layer_id, expert_id, slot });
    if (!pushed) {
        slot->state.store(SlotState::EMPTY, std::memory_order_release);
        return false;
    }
    return true;
}

RingSlot* Streamer::claim_ready() noexcept {
    Impl& im = *impl_;

    // Scan the ring for the first READY slot and transition it to CONSUMED.
    // O(ring_sz) — called once per expert execution step, fully acceptable.
    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        SlotState expected = SlotState::READY;
        if (im.slots[i].state.compare_exchange_strong(
                expected, SlotState::CONSUMED,
                std::memory_order_acquire,  // acquire: see all pread() writes
                std::memory_order_relaxed))
        {
            im.slots[i].last_used_tick.store(im.global_tick.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
            return &im.slots[i];
        }
    }
    return nullptr;
}

RingSlot* Streamer::claim_specific(uint32_t layer_id, uint32_t expert_id) noexcept {
    Impl& im = *impl_;
    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        if (im.slots[i].layer_id == layer_id && im.slots[i].expert_id == expert_id) {
            SlotState expected = SlotState::READY;
            if (im.slots[i].state.compare_exchange_strong(
                    expected, SlotState::CONSUMED,
                    std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                im.slots[i].ref_count.fetch_add(1, std::memory_order_relaxed);
                im.slots[i].last_used_tick.store(
                    im.global_tick.fetch_add(1, std::memory_order_relaxed),
                    std::memory_order_relaxed);
                return &im.slots[i];
            }

            // Retained-cache hit: release() leaves slots CONSUMED with their
            // data intact (only eviction/prune wipes it). Re-claiming an idle
            // populated slot is what makes the ring an actual LRU cache —
            // without it, prefetch()'s idempotency check ("already in ring")
            // plus a CONSUMED-only slot deadlocks the claim spin-loop.
            // Safe without CAS on state: eviction (inside prefetch) and
            // claiming both run on the caller thread only; workers only ever
            // transition LOADING→READY.
            if (expected == SlotState::CONSUMED &&
                im.slots[i].ref_count.load(std::memory_order_relaxed) == 0 &&
                im.slots[i].data_size > 0)
            {
                im.slots[i].ref_count.fetch_add(1, std::memory_order_relaxed);
                im.slots[i].last_used_tick.store(
                    im.global_tick.fetch_add(1, std::memory_order_relaxed),
                    std::memory_order_relaxed);
                return &im.slots[i];
            }
            // If slot is LOADING, it's being filled — return nullptr
            // so caller retries.
        }
    }
    return nullptr;
}

void Streamer::release(RingSlot* slot) noexcept {
    // Atomically decrement ref_count. DO NOT release to EMPTY!
    // The slot remains in CONSUMED/READY state so the LRU cache (prune_slots)
    // can retain it for future hits.
    slot->ref_count.fetch_sub(1, std::memory_order_acq_rel);
}

void Streamer::prune_slots(bool (*is_active)(uint32_t, uint32_t, void*), void* ctx) noexcept {
    Impl& im = *impl_;
    
    uint32_t empty_count = 0;
    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        if (im.slots[i].state.load(std::memory_order_relaxed) == SlotState::EMPTY) {
            empty_count++;
        }
    }
    
    // Always prune all inactive slots — the ring needs maximum available
    // capacity for the next token. With 94 MoE layers × 8 experts = 752
    // slots per token, any retention of stale slots risks deadlock.
    // The prefetch idempotency check prevents re-loading cached experts.
    uint32_t target_empty = im.ring_sz;  // free everything inactive
    if (empty_count >= target_empty) return;
    
    struct Candidate {
        uint32_t idx;
        uint64_t tick;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(im.ring_sz);

    for (uint32_t i = 0; i < im.ring_sz; ++i) {
        SlotState s = im.slots[i].state.load(std::memory_order_relaxed);
        if (s == SlotState::READY || s == SlotState::CONSUMED) {
            if (im.slots[i].ref_count.load(std::memory_order_relaxed) == 0) {
                if (!is_active(im.slots[i].layer_id, im.slots[i].expert_id, ctx)) {
                    candidates.push_back({i, im.slots[i].last_used_tick.load(std::memory_order_relaxed)});
                }
            }
        }
    }

    // Sort by oldest tick first
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.tick < b.tick;
    });

    for (const auto& c : candidates) {
        if (empty_count >= target_empty) break;
        uint32_t i = c.idx;
        im.slots[i].ref_count.store(0, std::memory_order_relaxed);
        im.slots[i].data_size = 0;
        im.slots[i].layer_id = 0xFFFFFFFF;
        im.slots[i].expert_id = 0xFFFFFFFF;
        im.slots[i].state.store(SlotState::EMPTY, std::memory_order_release);
        empty_count++;
    }
}

// ── Telemetry getters ─────────────────────────────────────────

uint64_t Streamer::bytes_read() const noexcept {
    return impl_->stat_bytes_read.load(std::memory_order_relaxed);
}

uint64_t Streamer::hit_count() const noexcept {
    return impl_->stat_hits.load(std::memory_order_relaxed);
}

uint64_t Streamer::miss_count() const noexcept {
    return impl_->stat_misses.load(std::memory_order_relaxed);
}

uint32_t Streamer::ring_size() const noexcept {
    return impl_->ring_sz;
}

uint32_t Streamer::ready_count() const noexcept {
    uint32_t count = 0;
    for (uint32_t i = 0; i < impl_->ring_sz; ++i) {
        if (impl_->slots[i].state.load(std::memory_order_relaxed) == SlotState::READY) {
            count++;
        }
    }
    return count;
}

const void* Streamer::pool_data() const noexcept { return impl_->data_pool; }
uint64_t Streamer::pool_size() const noexcept { return static_cast<uint64_t>(impl_->ring_sz) * impl_->slot_cap; }

const void* Streamer::get_slot_ptr(uint32_t index) const noexcept {
    if (index >= impl_->ring_sz) return nullptr;
    return impl_->slots[index].data;
}

uint64_t Streamer::get_slot_bytes() const noexcept {
    return impl_->slot_cap;
}

void Streamer::print_debug_states() const noexcept {
    uint32_t states[4] = {0};
    for (uint32_t i = 0; i < impl_->ring_sz; ++i) {
        SlotState s = impl_->slots[i].state.load(std::memory_order_relaxed);
        states[static_cast<int>(s)]++;
    }
    size_t eq = impl_->request_queue.get_enqueue_pos();
    size_t dq = impl_->request_queue.get_dequeue_pos();
    std::fprintf(stderr, "[DEBUG RING] EMPTY=%u LOADING=%u READY=%u CONSUMED=%u | QUEUE eq=%zu dq=%zu\n",
                 states[0], states[1], states[2], states[3], eq, dq);
    
    for (uint32_t i = 0; i < impl_->ring_sz; ++i) {
        SlotState s = impl_->slots[i].state.load(std::memory_order_relaxed);
        if (s != SlotState::EMPTY) {
            std::fprintf(stderr, "  Slot[%u]: state=%u L%u E%u ref=%u\n",
                         i, static_cast<uint32_t>(s),
                         impl_->slots[i].layer_id,
                         impl_->slots[i].expert_id,
                         impl_->slots[i].ref_count.load(std::memory_order_relaxed));
        }
    }
    std::fflush(stderr);
}

uint32_t Streamer::debug_slot_state(uint32_t layer_id, uint32_t expert_id) const noexcept {
    for (uint32_t i = 0; i < impl_->ring_sz; ++i) {
        if (impl_->slots[i].layer_id == layer_id && impl_->slots[i].expert_id == expert_id) {
            return static_cast<uint32_t>(impl_->slots[i].state.load(std::memory_order_relaxed));
        }
    }
    return 999;
}

void Streamer::debug_dump_ring() const noexcept {
    std::fprintf(stderr, "--- RING BUFFER DUMP ---\n");
    for (uint32_t i = 0; i < impl_->ring_sz; ++i) {
        uint32_t st = static_cast<uint32_t>(impl_->slots[i].state.load(std::memory_order_relaxed));
        if (st != 0) {
            std::fprintf(stderr, "Slot %u: layer=%u expert=%u state=%u ref_count=%u\n", 
                         i, impl_->slots[i].layer_id, impl_->slots[i].expert_id, st,
                         impl_->slots[i].ref_count.load(std::memory_order_relaxed));
        }
    }
    std::fprintf(stderr, "------------------------\n");
}

} // namespace smoe::io
