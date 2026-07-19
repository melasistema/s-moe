// ═══════════════════════════════════════════════════════════════
// engine.cpp — S-MoE Engine · Boot-Time Machinery
// ═══════════════════════════════════════════════════════════════
// See engine.hpp. Nothing here runs on the token hot path.
// ═══════════════════════════════════════════════════════════════

#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <mach/mach.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

namespace smoe::engine {

double wall_ms() noexcept {
    using namespace std::chrono;
    return double(duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

uint64_t resident_bytes() noexcept {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS)
    {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
}

// ── Slot / ring auto-tune ─────────────────────────────────────
//
// slot_bytes: must fit the largest expert blob in the vault. We scan
//   the expert table for the maximum padded_size and round up to 16 KB.
//   The user can override with --slot-mb.
//
// ring_size: subtractive budget — take the OS's own available-memory
//   estimate (kern.memorystatus_level, the signal `memory_pressure -Q`
//   prints) and reserve what the rest of the engine needs: the scout
//   mmap (fully resident after background prefault), a fixed engine
//   overhead, and an OS floor. Everything else becomes ring. The old
//   flat "25% of free+inactive" budget missed evictable file-backed
//   pages and left ~15 GB of a 48 GB machine idle.
//   The user can override with --ring.
void autotune(const char* vault_path, const char* scout_path,
              const SmoeHeader& hdr,
              uint64_t& slot_mb, uint32_t& ring_size)
{
    // ── 1. Slot size: scan expert table for max padded_size ──
    if (slot_mb == 0) {
        int scan_fd = ::open(vault_path, O_RDONLY);
        uint64_t max_padded = 0;
        if (scan_fd >= 0) {
            const size_t table_bytes = static_cast<size_t>(hdr.total_experts) * sizeof(smoe::ExpertEntry);
            std::vector<smoe::ExpertEntry> table(hdr.total_experts);
            ssize_t nr = ::pread(scan_fd, table.data(), table_bytes,
                                 static_cast<off_t>(hdr.table_offset));
            if (nr == static_cast<ssize_t>(table_bytes)) {
                for (const auto& e : table)
                    if (e.padded_size > max_padded) max_padded = e.padded_size;
            }
            ::close(scan_fd);
        }
        if (max_padded == 0) max_padded = 8ULL * 1024 * 1024; // fallback 8 MB
        // Round up to 16 KB boundary — use literal to avoid mach PAGE_SIZE macro collision
        constexpr uint64_t SMOE_PAGE = 16384ULL;
        uint64_t auto_slot = (max_padded + SMOE_PAGE - 1) & ~(SMOE_PAGE - 1);
        slot_mb = (auto_slot + (1024*1024 - 1)) / (1024*1024); // round up to MB
        if (slot_mb == 0) slot_mb = 1;
        std::fprintf(stderr, "[ring] Auto slot size: %llu MB  (max expert blob: %llu KB)\n",
            (unsigned long long)slot_mb,
            (unsigned long long)(max_padded / 1024));
    }

    // ── 2. Ring size: available RAM minus the engine's fixed costs ──
    if (ring_size == 0) {
        const uint64_t GB = 1024ULL * 1024 * 1024;

        // (a) Availability. Primary signal: kern.memorystatus_level —
        // the percentage of RAM the OS reports as usable before memory
        // pressure. Unlike free+inactive page counts it includes
        // evictable file-backed pages (e.g. a previous run's scout
        // mmap), which is most reclaimable memory on a warm machine.
        // Fallback: the old free+inactive scan.
        uint64_t vmstat_avail = 0;
        vm_statistics64_data_t vmstat {};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vmstat),
                              &count) == KERN_SUCCESS) {
            // free_count includes pages that are available without swapping
            uint64_t page_sz = static_cast<uint64_t>(vm_page_size);
            vmstat_avail = (static_cast<uint64_t>(vmstat.free_count) +
                            static_cast<uint64_t>(vmstat.inactive_count)) * page_sz;
        }
        if (vmstat_avail < 512ULL * 1024 * 1024)
            vmstat_avail = 512ULL * 1024 * 1024; // assume at least 512 MB

        uint64_t mem_total = 0;
        size_t   sc_len    = sizeof(mem_total);
        if (::sysctlbyname("hw.memsize", &mem_total, &sc_len, nullptr, 0) != 0)
            mem_total = 0;

        uint64_t avail_bytes = 0;
        uint32_t mem_level   = 0;
        sc_len = sizeof(mem_level);
        if (mem_total &&
            ::sysctlbyname("kern.memorystatus_level", &mem_level, &sc_len, nullptr, 0) == 0 &&
            mem_level > 0 && mem_level <= 100) {
            avail_bytes = mem_total / 100 * mem_level;
        }
        if (avail_bytes == 0) avail_bytes = vmstat_avail;

        // (b) Reserve what the ring must never take:
        //   - the scout file: background prefault makes it fully resident
        //   - fixed engine overhead: heavy KV cache (~1.6 GB at 4096 ctx)
        //     + activations + Metal scratch
        //   - an OS floor of 1/8 physical RAM (≥ 4 GB) so ring growth can
        //     never push the system into swap
        uint64_t scout_bytes = 0;
        if (scout_path) {
            struct stat sst {};
            if (::stat(scout_path, &sst) == 0)
                scout_bytes = static_cast<uint64_t>(sst.st_size);
        }
        uint64_t os_floor = mem_total ? std::max(mem_total / 8, 4 * GB) : 4 * GB;
        uint64_t reserve  = scout_bytes + 2 * GB + os_floor;

        // (c) Budget: subtractive, floored at the old 25%-of-free tuner
        // (never regress), capped at half of physical RAM (the pool is
        // wrapped in a single MTLBuffer — stay clear of device limits).
        uint64_t legacy_budget = vmstat_avail / 4;
        uint64_t ring_budget   = (avail_bytes > reserve) ? (avail_bytes - reserve)
                                                         : legacy_budget;
        if (ring_budget < legacy_budget) ring_budget = legacy_budget;
        if (mem_total && ring_budget > mem_total / 2) ring_budget = mem_total / 2;

        uint64_t slot_bytes_now = slot_mb * 1024ULL * 1024ULL;
        uint32_t auto_ring      = static_cast<uint32_t>(ring_budget / slot_bytes_now);

        // Clamp: minimum 64 (one full token's experts + spare), maximum 4096
        if (auto_ring < 64)   auto_ring = 64;
        if (auto_ring > 4096) auto_ring = 4096;
        ring_size = auto_ring;
        std::fprintf(stderr, "[ring] Auto ring size : %u slots × %llu MB = %.1f GB  (avail: %.1f GB, reserved: %.1f GB, level: %u%%)\n",
            ring_size, (unsigned long long)slot_mb,
            static_cast<double>(ring_size) * slot_mb / 1024.0,
            static_cast<double>(avail_bytes) / (1024.0*1024*1024),
            static_cast<double>(reserve) / (1024.0*1024*1024),
            mem_level);
    }
}

// ── Vocabulary ────────────────────────────────────────────────

namespace {
std::vector<std::string> g_vocab;   // sized at load; empty = not loaded
} // namespace

bool load_vocab(const char* path, uint32_t vocab_size, bool debug) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "[vocab] ⚠ Failed to open vocabulary file '%s'\n", path);
        return false;
    }
    g_vocab.assign(vocab_size, {});   // one-time boot allocation
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, f) != 1) break;
        g_vocab[i].resize(len);
        if (len > 0) {
            if (std::fread(&g_vocab[i][0], 1, len, f) != len) break;
        }
        loaded++;
    }
    std::fclose(f);
    if (loaded == 0) {
        std::fprintf(stderr, "[vocab] ✗ Failed to load any tokens\n");
        g_vocab.clear();
        return false;
    }
    if (debug) {
        std::fprintf(stderr, "[vocab] ✓ Loaded %u tokens from '%s'\n", loaded, path);
    }
    if (loaded < vocab_size) {
        std::fprintf(stderr, "[vocab] ⚠ Only loaded %u/%u tokens\n", loaded, vocab_size);
    }
    return true;
}

const std::string* vocab() noexcept {
    return g_vocab.empty() ? nullptr : g_vocab.data();
}

// ── Expert popularity histogram ───────────────────────────────

void load_expert_freq(uint32_t* hits) {
    FILE* f = std::fopen(EXPERT_FREQ_PATH, "rb");
    if (!f) return;
    uint32_t magic = 0, ver = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) == 1 && magic == 0x51524653u /* 'SFRQ' */ &&
        std::fread(&ver, 4, 1, f) == 1 && ver == 1 &&
        std::fread(&n, 4, 1, f) == 1 && n <= 128u * smoe::prefill::HITS_STRIDE) {
        static uint32_t tmp[128 * smoe::prefill::HITS_STRIDE];
        if (std::fread(tmp, 4, n, f) == n) {
            for (uint32_t i = 0; i < n; ++i) hits[i] += tmp[i];
        }
    }
    std::fclose(f);
}

void save_expert_freq(const uint32_t* hits) {
    FILE* f = std::fopen(EXPERT_FREQ_PATH, "wb");
    if (!f) return;
    uint32_t magic = 0x51524653u, ver = 1, n = 128u * smoe::prefill::HITS_STRIDE;
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&ver, 4, 1, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(hits, 4, n, f);
    std::fclose(f);
}

// ── Prewarm thread ────────────────────────────────────────────
// Ordering comes from the persisted claim histogram; unseen experts
// keep their sequential order so a first boot still warms
// deterministically. Joinable (not detached): a detached pass could
// outlive main's stack-local streamer on short runs and read a dead
// reference.

void Prewarm::start(smoe::io::Streamer* streamer, const uint32_t* hits,
                    uint32_t start_layer, uint32_t num_moe_layers,
                    uint32_t experts_per_layer)
{
    thread_ = std::thread([this, streamer, hits, start_layer,
                           num_moe_layers, experts_per_layer]() {
        constexpr uint32_t HS = smoe::prefill::HITS_STRIDE;
        static uint32_t order[128 * HS];
        uint32_t last_epoch = 0;
        while (!shutdown_.load(std::memory_order_relaxed)) {
            if (busy_.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            uint32_t epoch = epoch_.load(std::memory_order_relaxed);
            if (epoch == last_epoch) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            uint32_t n_entries = 0;
            for (uint32_t l = start_layer;
                 l < start_layer + num_moe_layers && l < 128; ++l) {
                for (uint32_t e = 0; e < experts_per_layer && e < HS; ++e) {
                    order[n_entries++] = l * HS + e;
                }
            }
            std::stable_sort(order, order + n_entries, [hits](uint32_t a, uint32_t b) {
                return hits[a] > hits[b];
            });

            // Reserve headroom for the live working set. Layer-major
            // prefill claims at most one layer's expert union (≤128
            // slots) plus in-flight fetches at a time, so max(128,
            // ring/4) suffices.
            const uint32_t ring_sz = streamer->ring_size();
            const uint32_t margin  = (ring_sz / 4 > 128) ? ring_sz / 4 : 128;
            uint32_t budget = (ring_sz > margin) ? ring_sz - margin : 0;
            for (uint32_t i = 0; i < n_entries && budget > 0; ++i) {
                if (shutdown_.load(std::memory_order_relaxed)) return;
                if (busy_.load(std::memory_order_relaxed)) break; // yield to request
                if (epoch_.load(std::memory_order_relaxed) != epoch) break;
                if (streamer->prefetch(order[i] / HS, order[i] % HS)) budget--;
                // Trickle: keep the MPMC queue shallow so demand
                // prefetches always find room the moment a request lands.
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            last_epoch = epoch;
        }
    });
}

void Prewarm::stop() noexcept {
    shutdown_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
}

// ── Buffers ───────────────────────────────────────────────────

void Buffers::init(SmoeMetalCtx* metal, const SmoeModelConfig& cfg,
                   const SmoeHeader& hdr, uint32_t attn_ctx)
{
    const uint32_t d_model    = cfg.d_model;
    const uint32_t ffn_dim    = cfg.ffn_dim;
    const uint32_t shared_dim = cfg.shared_expert_ffn_dim;
    const uint32_t q_dim      = cfg.num_heads * cfg.head_dim;
    const uint32_t kv_dim     = cfg.num_kv_heads * cfg.head_dim;
    const uint32_t rope_half  = cfg.head_dim / 2;
    const uint32_t num_layers = cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0);

    // One allocator for every plane: 16 KB-aligned (Metal zero-copy
    // wrapping), zero-filled, optionally registered with the GPU.
    auto plane = [&](size_t elems, bool reg) -> float* {
        void* p = nullptr;
        size_t bytes = (elems * sizeof(float) + 16383) & ~size_t(16383);
        if (::posix_memalign(&p, 16384, bytes) != 0 || !p) {
            std::fprintf(stderr, "  ✗  Buffer allocation failed (%zu bytes).\n", bytes);
            std::abort();
        }
        std::memset(p, 0, bytes);
        if (reg && metal) smoe_metal_register_buffer(metal, p, bytes);
        return static_cast<float*>(p);
    };

    // KV ring: one zero-copy wrap of the whole allocation so the fused
    // attention layer resolves per-layer cache slices as offsets into a
    // single registered buffer.
    kv_cache   = plane(size_t(num_layers) * 2 * attn_ctx * kv_dim, true);

    hidden     = plane(d_model, true);
    normed     = plane(d_model, true);
    qbuf       = plane(q_dim,   true);
    kbuf       = plane(kv_dim,  true);
    vbuf       = plane(kv_dim,  true);
    attn_out   = plane(q_dim,   true);
    routed_out = plane(d_model, false);
    shared_out = plane(d_model, false);
    // Per-expert FFN output rows fetched back from the GPU each layer.
    expert_out = plane(size_t(smoe::scout::MAX_ACTIVE) * d_model, false);
    l0_gate    = plane(ffn_dim, false);
    l0_up      = plane(ffn_dim, false);
    sh_gate    = (shared_dim > 0) ? plane(shared_dim, false) : nullptr;
    sh_up      = (shared_dim > 0) ? plane(shared_dim, false) : nullptr;
    gate_scores= plane(cfg.max_experts_per_layer, false);

    // RoPE: the rotation angle depends only on (position, dim) — not on
    // head or layer — so inv_freq is computed once and cos/sin once per
    // token step instead of per head × dim × layer.
    rope_inv_freq = plane(rope_half, false);
    rope_cos      = plane(rope_half, false);
    rope_sin      = plane(rope_half, false);
    for (uint32_t d = 0; d < rope_half; ++d) {
        rope_inv_freq[d] = 1.0f / std::pow(cfg.rope_theta,
            static_cast<float>(d * 2) / static_cast<float>(cfg.head_dim));
    }

    // Layer-major prefill activation planes (one row per chunk token).
    {
        using smoe::prefill::CHUNK;
        pf.hidden      = plane(size_t(CHUNK) * d_model, false);
        pf.normed      = plane(size_t(CHUNK) * d_model, true);   // matvec + fused_ffn input rows
        pf.qbuf        = plane(size_t(CHUNK) * q_dim,  true);
        pf.kbuf        = plane(size_t(CHUNK) * kv_dim, true);
        pf.vbuf        = plane(size_t(CHUNK) * kv_dim, true);
        pf.attn_out    = plane(size_t(CHUNK) * q_dim,  true);
        pf.routed_out  = plane(size_t(CHUNK) * d_model, false);
        pf.oproj_out   = plane(size_t(CHUNK) * d_model, true);  // one o_proj row per token (batched dispatch)
        pf.ffn_hidden  = plane(ffn_dim, true);
        pf.ffn_up      = plane(ffn_dim, true);
        pf.ffn_out     = plane(d_model, true);
        pf.sh_gate     = (shared_dim > 0) ? plane(shared_dim, true) : nullptr;
        pf.sh_up       = (shared_dim > 0) ? plane(shared_dim, true) : nullptr;
        pf.sh_out      = (shared_dim > 0) ? plane(d_model, true) : nullptr;
        pf.gate_scores = plane(hdr.max_experts_per_layer, false);
        pf.rope_cos    = plane(size_t(CHUNK) * rope_half, false);
        pf.rope_sin    = plane(size_t(CHUNK) * rope_half, false);
        pf.batch_in    = plane(size_t(CHUNK) * d_model, false);
        pf.batch_out   = plane(size_t(CHUNK) * d_model, false);
    }
}

} // namespace smoe::engine
