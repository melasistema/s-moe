// ═══════════════════════════════════════════════════════════════
// main.cpp — S-MoE Engine · Core Loop & Live Telemetry
// ═══════════════════════════════════════════════════════════════
// Phase 4 — Week 4
//
// Responsibilities:
//   ① CLI argument parsing and validation
//   ② Pre-flight vault inspection (SmoeHeader read + magic check)
//   ③ Token generation loop:
//        Scout.forward() → prefetch() → Metal execute → emit token
//   ④ Live telemetry bar: t/s | RAM | NVMe GB/s | miss %
//   ⑤ Clean shutdown: drain ring, join workers, free Metal
//
// Design invariants (GEMINI.md):
//   ① Zero heap allocations inside the token generation loop.
//   ② All synchronisation via atomics in Streamer and MetalCtx.
//   ③ Telemetry reads are lock-free (atomic loads, no mutex).
// ═══════════════════════════════════════════════════════════════

#include "common.hpp"
#include "io/streamer.hpp"
#include "scout/scout.hpp"
#include "compute/metal_bridge.h"
#include "cli.hpp"
#include "sampler.hpp"
#include "prefill.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <thread>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>

// ── Compile-time defaults ─────────────────────────────────────
inline constexpr uint32_t TELEMETRY_EVERY      = 10;   // tokens between telemetry updates

// ── Telemetry helpers ─────────────────────────────────────────

// Returns the process resident set size in bytes via macOS task_info.
static uint64_t resident_bytes() noexcept {
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

// Wall-clock time in milliseconds since an arbitrary epoch.
static double wall_ms() noexcept {
    using namespace std::chrono;
    return double(duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

// ── Telemetry bar ─────────────────────────────────────────────

struct TelemetryState {
    double   start_ms    { 0.0 };
    uint64_t prev_bytes  { 0 };
    double   prev_ms     { 0.0 };
};

// ── Decode instrumentation (--instrument) ─────────────────────
// Wall-time buckets over the serial generation loop, accumulated per
// request and printed at request end. Buckets are contiguous segments
// between marks on the main thread, so their sum plus "other" equals
// the measured token total — no double counting.
struct InstrState {
    double   prune_ms    { 0.0 };  // prune_slots (both call sites)
    double   dense_ms    { 0.0 };  // norms + QKV + attention + o_proj (+ shared)
    double   dispatch_ms { 0.0 };  // Phase 1 first claim + GPU dispatch pass
    double   io_spin_ms  { 0.0 };  // Phase 3 spin-wait on missing experts
    double   gpu_wait_ms { 0.0 };  // Phase 4 kernel wait + accumulate
    double   lm_ms       { 0.0 };  // final norm + LM head + sampling
    double   total_ms    { 0.0 };
    uint64_t tokens      { 0 };
    uint64_t bytes       { 0 };
    uint64_t ovl_hits    { 0 };    // experts shared with the previous token
    uint64_t ovl_total   { 0 };    // experts routed this token
    uint64_t cov[4]      { 0, 0, 0, 0 };  // true experts inside prev token's top-8/16/24/32
    uint64_t cov_total   { 0 };
    void reset() { *this = InstrState{}; }
};

// Previous token's routed-expert bitmask per layer, for the adjacent-
// token overlap measurement (≤128 experts → two 64-bit words).
static uint64_t g_instr_prev_mask[128][2];

// Previous token's top-32 gate ranking per layer: measures how much of
// the next token's true top-8 a top-M speculative prefetch would cover.
static uint32_t g_instr_prev_top32[128][32];
static uint8_t  g_instr_prev_top32_n[128];

static void print_telemetry(
    uint32_t               token_n,
    const smoe::io::Streamer& streamer,
    TelemetryState&        ts)
{
    double   now_ms      = wall_ms();
    double   elapsed_s   = (now_ms - ts.start_ms) / 1000.0;
    double   tps         = (elapsed_s > 0.0) ? double(token_n) / elapsed_s : 0.0;

    // NVMe GB/s (delta since last telemetry update)
    uint64_t cur_bytes   = streamer.bytes_read();
    double   dt_s        = (now_ms - ts.prev_ms) / 1000.0;
    double   gbps        = (dt_s > 0.0)
                           ? double(cur_bytes - ts.prev_bytes) / 1e9 / dt_s
                           : 0.0;
    ts.prev_bytes = cur_bytes;
    ts.prev_ms    = now_ms;

    // RAM footprint
    double   ram_gb      = double(resident_bytes()) / 1e9;

    // Expert miss rate
    uint64_t hits   = streamer.hit_count();
    uint64_t misses = streamer.miss_count();
    double   miss_pct = (hits + misses > 0)
                        ? 100.0 * double(misses) / double(hits + misses)
                        : 0.0;

    // \r to overwrite the same line in the terminal
    std::fprintf(stderr,
        "\r  S-MoE ▸  %5.1f t/s  │  RAM %4.1f GB  │  NVMe %4.1f GB/s  │  miss %4.1f%%   ",
        tps, ram_gb, gbps, miss_pct);
    std::fflush(stderr);
}

// ── Vault pre-flight ──────────────────────────────────────────

static bool read_vault_header(const char* vault_path, smoe::SmoeHeader& hdr) {
    int fd = ::open(vault_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "\n  ✗  Cannot open vault: %s\n", vault_path);
        return false;
    }

    ssize_t n = ::read(fd, &hdr, sizeof(smoe::SmoeHeader));
    ::close(fd);

    if (n != ssize_t(sizeof(smoe::SmoeHeader))) {
        std::fprintf(stderr, "\n  ✗  Vault too small (not a valid .smoe file).\n");
        return false;
    }
    if (!smoe::magic_valid(hdr.magic)) {
        std::fprintf(stderr, "\n  ✗  Invalid magic bytes — not a .smoe vault.\n");
        return false;
    }
    if (hdr.version != smoe::SMOE_VERSION) {
        std::fprintf(stderr,
            "\n  ✗  Vault version mismatch: got %u, expected %u.\n",
            hdr.version, smoe::SMOE_VERSION);
        return false;
    }
    return true;
}

using smoe::prefill::ExpertLayout;

static bool read_expert_layout(const char* vault_path, const smoe::SmoeHeader& hdr, ExpertLayout& layout) {
    int fd = ::open(vault_path, O_RDONLY);
    if (fd < 0) return false;

    // TensorDescriptor table starts after the ExpertTable
    uint64_t table_bytes = static_cast<uint64_t>(hdr.total_experts) * sizeof(smoe::ExpertEntry);
    uint64_t desc_base = hdr.table_offset + table_bytes;

    smoe::TensorDescriptor descs[3];
    ssize_t n = ::pread(fd, descs, sizeof(descs), desc_base);
    ::close(fd);

    if (n != sizeof(descs)) return false;

    // Fill the layout
    layout.gate_rows = descs[0].rows;
    layout.gate_cols = descs[0].cols;
    layout.gate_packed_offset = descs[0].packed_offset;
    layout.gate_scales_offset = descs[0].scales_offset;

    layout.up_rows = descs[1].rows;
    layout.up_cols = descs[1].cols;
    layout.up_packed_offset = descs[1].packed_offset;
    layout.up_scales_offset = descs[1].scales_offset;

    layout.down_rows = descs[2].rows;
    layout.down_cols = descs[2].cols;
    layout.down_packed_offset = descs[2].packed_offset;
    layout.down_scales_offset = descs[2].scales_offset;

    return true;
}

static bool g_vocab_loaded = false;
static std::string g_vocab[151936];
static bool g_debug = false;
static bool g_raw_ids = false;
static bool g_instr = false;

// ── Expert popularity histogram (prewarm ordering) ────────────
// Claim counts per (absolute layer, expert), persisted across runs so
// the prewarm thread can stream the historically hottest experts first.
static uint32_t g_expert_hits[128 * smoe::prefill::HITS_STRIDE];

static void load_expert_freq(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return;
    uint32_t magic = 0, ver = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) == 1 && magic == 0x51524653u /* 'SFRQ' */ &&
        std::fread(&ver, 4, 1, f) == 1 && ver == 1 &&
        std::fread(&n, 4, 1, f) == 1 && n <= 128u * smoe::prefill::HITS_STRIDE) {
        static uint32_t tmp[128 * smoe::prefill::HITS_STRIDE];
        if (std::fread(tmp, 4, n, f) == n) {
            for (uint32_t i = 0; i < n; ++i) g_expert_hits[i] += tmp[i];
        }
    }
    std::fclose(f);
}

static void save_expert_freq(const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    uint32_t magic = 0x51524653u, ver = 1, n = 128u * smoe::prefill::HITS_STRIDE;
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&ver, 4, 1, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(g_expert_hits, 4, n, f);
    std::fclose(f);
}

static void load_vocab(const char* vocab_bin_path, uint32_t expected_vocab_size) {
    FILE* f = std::fopen(vocab_bin_path, "rb");
    if (!f) {
        std::fprintf(stderr, "[vocab] ⚠ Failed to open vocabulary file '%s'\n", vocab_bin_path);
        return;
    }
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < expected_vocab_size; ++i) {
        uint32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, f) != 1) break;
        g_vocab[i].resize(len);
        if (len > 0) {
            if (std::fread(&g_vocab[i][0], 1, len, f) != len) break;
        }
        loaded++;
    }
    std::fclose(f);
    if (loaded > 0) {
        g_vocab_loaded = true;
        if (g_debug) {
            std::fprintf(stderr, "[vocab] ✓ Loaded %u tokens from '%s'\n", loaded, vocab_bin_path);
        }
        if (loaded < expected_vocab_size) {
            std::fprintf(stderr, "[vocab] ⚠ Only loaded %u/%u tokens\n", loaded, expected_vocab_size);
        }
    } else {
        std::fprintf(stderr, "[vocab] ✗ Failed to load any tokens\n");
    }
}

// ── Token generation loop ─────────────────────────────────────

// Simple tokeniser shim: maps prompt string → token IDs.
// Week 4: one token per character (ASCII ordinal), capped at 65535.
// Week 5+: replace with a real BPE tokeniser.
static void tokenise_prompt(
    const char*  prompt,
    uint32_t*    token_ids,
    uint32_t&    count,
    uint32_t     max_tokens)
{
    count = 0;
    for (const char* p = prompt; *p && count < max_tokens; ++p) {
        token_ids[count++] = static_cast<uint32_t>(static_cast<unsigned char>(*p));
    }
}

static std::string decode_token(uint32_t id) {
    if (g_raw_ids) return "[" + std::to_string(id) + "] ";
    if (id < 151936 && !g_vocab[id].empty()) return g_vocab[id];
    return "<unk>";
}

// ── Main entry point ──────────────────────────────────────────

int main(int argc, char* argv[]) {

    // ── Argument parsing ──────────────────────────────────────
    smoe::cli::EngineConfig cli_cfg = smoe::cli::parse_args(argc, argv);
    if (!cli_cfg.valid) return 1;

    const char* vault_path  = cli_cfg.vault_path;
    const char* scout_path  = cli_cfg.scout_path;
    const char* prompt_text = cli_cfg.prompt_text;
    const char* tokens_in   = cli_cfg.tokens_in;
    uint32_t    max_tokens  = cli_cfg.max_tokens;
    uint32_t    ring_size   = cli_cfg.ring_size;
    uint32_t    num_workers = cli_cfg.num_workers;
    uint64_t    slot_mb     = cli_cfg.slot_mb;
    float       temperature = cli_cfg.temperature;
    float       top_p       = cli_cfg.top_p;
    uint32_t    top_k       = cli_cfg.top_k;
    float       rep_penalty = cli_cfg.rep_penalty;
    std::vector<uint32_t> eos_ids = cli_cfg.eos_ids;
    g_debug = cli_cfg.debug;
    g_raw_ids = cli_cfg.raw_ids;
    g_instr = cli_cfg.instrument;

    // ── Vault header read ────────────────────────────────────────────────────────────────────────
    smoe::SmoeHeader vault_hdr {};
    if (!read_vault_header(vault_path, vault_hdr)) return 1;

    if (g_debug) {
        std::fprintf(stderr,
            "\n"
            "  S-MoE Engine — Week 4 Heuristic Build\n"
            "  ───────────────────────────────────────\n"
            "  Vault      : %s\n"
            "  MoE layers : %u\n"
            "  Experts/L  : %u  (total: %u)\n"
            "  Ring slots : %u × %llu MB\n"
            "  Workers    : %u\n"
            "\n",
            vault_path,
            vault_hdr.num_moe_layers,
            vault_hdr.max_experts_per_layer,
            vault_hdr.total_experts,
            ring_size, slot_mb,
            num_workers);
    }

    ExpertLayout expert_layout {};
    if (!read_expert_layout(vault_path, vault_hdr, expert_layout)) {
        std::fprintf(stderr, "  ✗  Failed to read expert layout descriptors from vault.\n");
        return 1;
    }

    // ── Auto-compute slot size and ring size from vault + available RAM ──
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
    //   Minimum ring = 8 × num_active_experts (8 per layer) = 64 slots minimum.
    //   The user can override with --ring.
    {
        // ── 1. Slot size: scan expert table for max padded_size ──
        if (slot_mb == 0) {
            int scan_fd = ::open(vault_path, O_RDONLY);
            uint64_t max_padded = 0;
            if (scan_fd >= 0) {
                const size_t table_bytes = static_cast<size_t>(vault_hdr.total_experts) * sizeof(smoe::ExpertEntry);
                std::vector<smoe::ExpertEntry> table(vault_hdr.total_experts);
                ssize_t nr = ::pread(scan_fd, table.data(), table_bytes,
                                     static_cast<off_t>(vault_hdr.table_offset));
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
            //   - fixed engine overhead: KV cache + activations + Metal scratch
            //   - an OS floor of 1/8 physical RAM (≥ 4 GB) so ring growth can
            //     never push the system into swap
            uint64_t scout_bytes = 0;
            if (scout_path) {
                struct stat sst {};
                if (::stat(scout_path, &sst) == 0)
                    scout_bytes = static_cast<uint64_t>(sst.st_size);
            }
            uint64_t os_floor = mem_total ? std::max(mem_total / 8, 4 * GB) : 4 * GB;
            uint64_t reserve  = scout_bytes + 3 * GB + os_floor;

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

    uint64_t slot_bytes = slot_mb * 1024ULL * 1024ULL;

    // Phase 3: Metal bridge init ────────────────────────────────
    SmoeMetalCtx* metal = smoe_metal_init(slot_bytes);
    if (!metal) {
        std::fprintf(stderr, "  ✗  Metal initialisation failed.\n");
        return 1;
    }

    // ── Phase 2: Streamer init ────────────────────────────────
    smoe::io::Streamer streamer(vault_path, ring_size, num_workers, slot_bytes);
    smoe_metal_register_buffer(metal, streamer.pool_data(), streamer.pool_size());

    // ── Phase 4: Scout init ───────────────────────────────────
    smoe::scout::Scout scout(scout_path, metal, &vault_hdr);  // scout_path may be nullptr (heuristic only)
    const auto& cfg = scout.config();
    const uint32_t num_layers = cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0);

    // ── Popularity-Ordered Cache Pre-Warming ──────────────────
    // Streams the historically hottest experts into the ring while the
    // engine is idle: at startup and, in serve mode, between requests
    // (bump prewarm_epoch to re-arm a pass). engine_busy pauses it so
    // it never competes with demand reads. Ordering comes from the
    // persisted claim histogram; unseen experts keep their sequential
    // order so a first boot still warms deterministically.
    // Joinable (not detached): a detached pass could outlive main's
    // stack-local streamer on short runs and read a dead reference.
    load_expert_freq("vault/expert_freq.bin");
    std::atomic<bool>     prewarm_shutdown { false };
    std::atomic<bool>     engine_busy      { false };
    std::atomic<uint32_t> prewarm_epoch    { 1 };
    const uint32_t prewarm_start_layer = cfg.has_dense_layer_0 ? 1 : 0;
    std::thread prewarm_thread([&]() {
        constexpr uint32_t HS = smoe::prefill::HITS_STRIDE;
        static uint32_t order[128 * HS];
        uint32_t last_epoch = 0;
        while (!prewarm_shutdown.load(std::memory_order_relaxed)) {
            if (engine_busy.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            uint32_t epoch = prewarm_epoch.load(std::memory_order_relaxed);
            if (epoch == last_epoch) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            uint32_t n_entries = 0;
            for (uint32_t l = prewarm_start_layer;
                 l < prewarm_start_layer + cfg.num_moe_layers && l < 128; ++l) {
                for (uint32_t e = 0; e < vault_hdr.max_experts_per_layer && e < HS; ++e) {
                    order[n_entries++] = l * HS + e;
                }
            }
            std::stable_sort(order, order + n_entries, [](uint32_t a, uint32_t b) {
                return g_expert_hits[a] > g_expert_hits[b];
            });

            // Reserve headroom for the live working set. Layer-major
            // prefill claims at most one layer's expert union (≤128
            // slots) plus in-flight fetches at a time, so max(128,
            // ring/4) suffices. (The old "ring − 800" margin was sized
            // for a per-token working set that no longer exists — and
            // made prewarm a silent no-op on rings under 800 slots,
            // e.g. 580 here on 32/48 GB machines.)
            const uint32_t ring_sz = streamer.ring_size();
            const uint32_t margin  = (ring_sz / 4 > 128) ? ring_sz / 4 : 128;
            uint32_t budget = (ring_sz > margin) ? ring_sz - margin : 0;
            for (uint32_t i = 0; i < n_entries && budget > 0; ++i) {
                if (prewarm_shutdown.load(std::memory_order_relaxed)) return;
                if (engine_busy.load(std::memory_order_relaxed)) break; // yield to request
                if (prewarm_epoch.load(std::memory_order_relaxed) != epoch) break;
                if (streamer.prefetch(order[i] / HS, order[i] % HS)) budget--;
                // Trickle: keep the MPMC queue shallow so demand
                // prefetches always find room the moment a request lands.
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            last_epoch = epoch;
        }
    });

    const uint32_t d_model = cfg.d_model;
    const uint32_t ffn_dim = cfg.ffn_dim;
    const uint32_t shared_dim = cfg.shared_expert_ffn_dim; // 0 = no shared expert (Qwen3-235B), >0 = DeepSeek
    const uint32_t moe_start_layer = cfg.has_dense_layer_0 ? 1 : 0;


    // ── Load vocabulary ───────────────────────────────────────
    load_vocab("vault/vocab.bin", cfg.vocab_size);

    // ── Persistent token stream (server mode) ─────────────────
    // stream_tokens is the engine's canonical view of the whole
    // conversation (prompt + generated), used for (a) repetition-penalty
    // history and (b) longest-common-prefix matching so a follow-up
    // request only prefills its new suffix against the retained KV cache.
    static constexpr uint32_t STREAM_CAP = 8192;
    static uint32_t stream_tokens[STREAM_CAP];
    uint32_t stream_len = 0;

    // Per-request prompt suffix. Pre-allocated on the stack — no heap in the loop.
    uint32_t* prompt_tokens = (uint32_t*)__builtin_alloca(4096 * sizeof(uint32_t));
    uint32_t  prompt_len = 0;

    std::mt19937 rng(1337); // Fixed seed for deterministic debug, can be randomized later
    // Top-K candidate buffer for the sampler — allocated once here so the
    // token loop stays heap-free.
    smoe::SamplerScratch sampler_scratch(top_k, cfg.vocab_size);
    TelemetryState ts;

    // Instrumentation marks: instr_bucket() charges the wall time since
    // the previous mark to one bucket and advances the mark. No-op when
    // the current step is not an instrumented generation step.
    InstrState instr {};
    bool   instr_gen = false;
    double it_mark   = 0.0;
    auto instr_bucket = [&](double& acc) {
        if (!instr_gen) return;
        const double now = wall_ms();
        acc += now - it_mark;
        it_mark = now;
    };

    // ── Token Generation Loop ─────────────────────────────────
    //
    // Invariant per token step N:
    //
    //   1. Scout runs ahead up to K steps, queueing its predicted tokens
    //      and routing gates, firing prefetch requests.
    //   2. Heavy Model executes one step.
    //   3. Compare Heavy token vs Scout token.
    //   4. If mismatch, rollback Scout KV-cache and flush queue.
    //   5. Every TELEMETRY_EVERY tokens: update telemetry bar

    // Multi-layer KV cache (static to avoid heap allocation)
    static constexpr uint32_t ATTN_CTX = 4096;
    const uint32_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    const uint32_t q_dim = cfg.num_heads * cfg.head_dim;
    
    // Allocate full KV cache for all layers: layer -> (K, V) -> slot -> kv_dim
    float* full_kv_cache = smoe::allocate_aligned_float((cfg.num_moe_layers + (cfg.has_dense_layer_0 ? 1 : 0)) * 2 * ATTN_CTX * kv_dim);
    uint32_t ctx_pos = 0;
    uint32_t ctx_fill = 0;


    // ── Pre-allocate and register Heavy Execution Buffers ───────
    float* heavy_hidden = nullptr;
    float* heavy_normed = nullptr;
    float* heavy_qbuf = nullptr;
    float* heavy_kbuf = nullptr;
    float* heavy_vbuf = nullptr;
    float* heavy_attn_out = nullptr;
    float* routed_out = smoe::allocate_aligned_float(d_model);
    float* shared_out = smoe::allocate_aligned_float(d_model);
    
    // Allocate all on 16KB boundaries to guarantee Metal zero-copy wrapping works perfectly
    size_t heavy_hidden_aligned = (d_model * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_hidden, 16384, heavy_hidden_aligned);
    size_t heavy_normed_aligned = (d_model * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_normed, 16384, heavy_normed_aligned);
    size_t qbuf_aligned = (q_dim * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_qbuf,   16384, qbuf_aligned);
    size_t kvbuf_aligned = (kv_dim * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_kbuf,   16384, kvbuf_aligned);
    ::posix_memalign((void**)&heavy_vbuf,   16384, kvbuf_aligned);
    size_t heavy_attn_out_aligned = (q_dim * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_attn_out, 16384, heavy_attn_out_aligned);
    
    // Allocate heuristic dense/shared FFN intermediate buffers
    float* l0_gate_out = smoe::allocate_aligned_float(ffn_dim);
    float* l0_up_out   = smoe::allocate_aligned_float(ffn_dim);
    float* shared_gate_out = smoe::allocate_aligned_float(shared_dim);
    float* shared_up_out   = smoe::allocate_aligned_float(shared_dim);

    // Exact routing scratch: at every step (prompt AND generation) the
    // router gate is evaluated directly on the heavy hidden state
    // (128×d_model matvec, ~0.5M MACs) — the true Qwen3 routing. Ranks
    // beyond 8 (up to --spec extra) double as the next-token
    // speculative prefetch.
    const uint32_t spec_width = cli_cfg.spec_width;
    float* exact_gate_scores = smoe::allocate_aligned_float(vault_hdr.max_experts_per_layer);
    smoe::scout::ExpertPrediction exact_pred {};
    static uint32_t exact_rank_ids[32];
    static float    exact_rank_w[32];
    // NOTE: replaying the previous token's routing as PREFILL prefetch
    // hints was tried and measured SLOWER (114s vs 84s TTFT on the
    // 45-token bench) — hints added wrong reads ahead of demand fetches.
    // Prefill I/O overlap comes from layer-major batched prefill instead.
    // Decode is different: measured adjacent-token expert overlap there
    // is ~53%, which ring retention exploits for free (it skips reads
    // rather than adding them).

    // RoPE tables: the rotation angle depends only on (position, dim) —
    // not on head or layer — so inv_freq is computed once and cos/sin
    // once per token step instead of per head × dim × layer.
    const uint32_t rope_half = cfg.head_dim / 2;
    float* rope_inv_freq = smoe::allocate_aligned_float(rope_half);
    float* rope_cos      = smoe::allocate_aligned_float(rope_half);
    float* rope_sin      = smoe::allocate_aligned_float(rope_half);
    for (uint32_t d = 0; d < rope_half; ++d) {
        rope_inv_freq[d] = 1.0f / std::pow(cfg.rope_theta,
            static_cast<float>(d * 2) / static_cast<float>(cfg.head_dim));
    }

    smoe_metal_register_buffer(metal, heavy_hidden, heavy_hidden_aligned);
    smoe_metal_register_buffer(metal, heavy_normed, heavy_normed_aligned);
    smoe_metal_register_buffer(metal, heavy_qbuf,   qbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_kbuf,   kvbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_vbuf,   kvbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_attn_out, heavy_attn_out_aligned);

    // ── Layer-major prefill activation planes ─────────────────
    // One row per chunk token. 16 KB-aligned so Metal can wrap them
    // zero-copy; registered once here, reused for every request.
    smoe::prefill::Buffers pf {};
    {
        using smoe::prefill::CHUNK;
        auto plane = [&](size_t elems, bool reg) -> float* {
            void* p = nullptr;
            size_t bytes = (elems * sizeof(float) + 16383) & ~size_t(16383);
            if (::posix_memalign(&p, 16384, bytes) != 0 || !p) return nullptr;
            std::memset(p, 0, bytes);
            if (reg) smoe_metal_register_buffer(metal, p, bytes);
            return static_cast<float*>(p);
        };
        pf.hidden      = plane(size_t(CHUNK) * d_model, false);
        pf.normed      = plane(size_t(CHUNK) * d_model, true);   // matvec + fused_ffn input rows
        pf.qbuf        = plane(size_t(CHUNK) * q_dim,  true);
        pf.kbuf        = plane(size_t(CHUNK) * kv_dim, true);
        pf.vbuf        = plane(size_t(CHUNK) * kv_dim, true);
        pf.attn_out    = plane(size_t(CHUNK) * q_dim,  true);
        pf.routed_out  = plane(size_t(CHUNK) * d_model, false);
        pf.oproj_out   = plane(d_model, true);
        pf.ffn_hidden  = plane(ffn_dim, true);
        pf.ffn_up      = plane(ffn_dim, true);
        pf.ffn_out     = plane(d_model, true);
        pf.sh_gate     = (shared_dim > 0) ? plane(shared_dim, true) : nullptr;
        pf.sh_up       = (shared_dim > 0) ? plane(shared_dim, true) : nullptr;
        pf.sh_out      = (shared_dim > 0) ? plane(d_model, true) : nullptr;
        pf.gate_scores = plane(vault_hdr.max_experts_per_layer, false);
        pf.rope_cos    = plane(size_t(CHUNK) * rope_half, false);
        pf.rope_sin    = plane(size_t(CHUNK) * rope_half, false);
        pf.batch_in    = plane(size_t(CHUNK) * d_model, false);
        pf.batch_out   = plane(size_t(CHUNK) * d_model, false);
    }

    // NOTE: the decode-time scout speculation (Phase A lookahead queue +
    // divergence rollback) was removed after measurement: the scout's
    // full backbone pass cost ~265 ms/token, and its routing predictions
    // matched the true exact-gate routing only 51.5% of the time —
    // barely above the 46.4% the retained ring already provides for
    // free. The decode prefetch oracle is now the previous token's own
    // top-16 gate ranking (measured 63.7% coverage of the next token's
    // true top-8), fired through the streamer's low-priority queue.
    uint32_t next_heavy_token = 0;

    const bool serve = cli_cfg.serve;
    bool one_shot_done = false;
    std::string req_line; // reused across requests; allocation stays out of the token loop

    // ═══ REQUEST LOOP ═══════════════════════════════════════════
    // One-shot mode runs exactly one request from the CLI arguments.
    // Server mode reads one request per stdin line:
    //   GEN <max_tokens> <id0>,<id1>,...   full conversation token stream
    //   RESET                              drop all cached state
    // and replies with raw token IDs on stdout, terminated by <<DONE>>.
    // The KV cache, ring cache, and scout state persist across requests;
    // only the suffix beyond the longest common prefix with the stored
    // stream is prefetched and prefilled.
    while (true) {
        uint32_t req_max_tokens = max_tokens;
        prompt_len = 0;

        if (!serve) {
            if (one_shot_done) break;
            one_shot_done = true;
            if (tokens_in) {
                std::string s(tokens_in);
                size_t pos = 0;
                while (pos < s.size() && prompt_len < 4096) {
                    size_t next_comma = s.find(',', pos);
                    if (next_comma == std::string::npos) {
                        prompt_tokens[prompt_len++] = std::stoul(s.substr(pos));
                        break;
                    }
                    prompt_tokens[prompt_len++] = std::stoul(s.substr(pos, next_comma - pos));
                    pos = next_comma + 1;
                }
            } else {
                tokenise_prompt(prompt_text, prompt_tokens, prompt_len,
                                std::min(max_tokens, uint32_t(4096)));
            }
            for (uint32_t i = 0; i < prompt_len && stream_len < STREAM_CAP; ++i) {
                stream_tokens[stream_len++] = prompt_tokens[i];
            }
        } else {
            if (!std::getline(std::cin, req_line)) break; // EOF → clean exit
            if (req_line.empty()) continue;

            if (req_line == "RESET") {
                ctx_pos = 0; ctx_fill = 0;
                stream_len = 0;
                scout.reset_context();
                std::printf("<<DONE>>\n");
                std::fflush(stdout);
                continue;
            }
            if (req_line.rfind("GEN ", 0) != 0) {
                std::printf("<<ERR bad_request>>\n");
                std::fflush(stdout);
                continue;
            }

            // Parse: GEN <max_tokens> <csv-ids>
            static uint32_t full_toks[STREAM_CAP];
            uint32_t full_len = 0;
            bool parse_ok = true;
            {
                size_t p = 4;
                size_t sp = req_line.find(' ', p);
                if (sp == std::string::npos) { parse_ok = false; }
                else {
                    req_max_tokens = static_cast<uint32_t>(std::atoi(req_line.c_str() + p));
                    if (req_max_tokens == 0 || req_max_tokens > 8192) parse_ok = false;
                    p = sp + 1;
                    while (parse_ok && p < req_line.size()) {
                        if (full_len >= STREAM_CAP) { parse_ok = false; break; }
                        char* endp = nullptr;
                        unsigned long v = std::strtoul(req_line.c_str() + p, &endp, 10);
                        if (endp == req_line.c_str() + p) { parse_ok = false; break; }
                        full_toks[full_len++] = static_cast<uint32_t>(v);
                        p = static_cast<size_t>(endp - req_line.c_str());
                        if (p < req_line.size() && req_line[p] == ',') ++p;
                    }
                }
            }
            if (!parse_ok || full_len == 0) {
                std::printf("<<ERR %s>>\n", full_len >= STREAM_CAP ? "ctx_overflow" : "bad_request");
                std::fflush(stdout);
                continue;
            }

            // Longest common prefix against the stored stream: matching
            // positions already live in the KV cache and are skipped.
            uint32_t lcp = 0;
            while (lcp < stream_len && lcp < full_len &&
                   stream_tokens[lcp] == full_toks[lcp]) ++lcp;

            if (lcp < stream_len) {
                // History was edited/regenerated — cached KV beyond the
                // divergence point is invalid. Full reset, full prefill.
                ctx_pos = 0; ctx_fill = 0;
                stream_len = 0;
                scout.reset_context();
                lcp = 0;
            }

            uint32_t suffix = full_len - lcp;
            if (suffix == 0) {
                std::printf("<<ERR empty_suffix>>\n");
                std::fflush(stdout);
                continue;
            }
            if (suffix > 4096) {
                std::printf("<<ERR suffix_too_long>>\n");
                std::fflush(stdout);
                continue;
            }
            for (uint32_t i = 0; i < suffix; ++i) {
                prompt_tokens[i] = full_toks[lcp + i];
            }
            prompt_len = suffix;
            for (uint32_t i = 0; i < prompt_len && stream_len < STREAM_CAP; ++i) {
                stream_tokens[stream_len++] = prompt_tokens[i];
            }
            if (g_debug) {
                std::fprintf(stderr, "[serve] stream=%u lcp=%u suffix=%u gen=%u\n",
                             full_len, lcp, suffix, req_max_tokens);
            }
        }

        // Absolute position of prompt_tokens[0] in the token stream —
        // RoPE and the scout sync must use stream positions, not the
        // per-request loop index.
        const uint64_t stream_base = stream_len - prompt_len;

        // Pause the prewarm trickle for the duration of the request so
        // demand reads own the NVMe queue.
        engine_busy.store(true, std::memory_order_relaxed);

        // ── Per-request state ─────────────────────────────────
        ts.start_ms   = wall_ms();
        ts.prev_ms    = ts.start_ms;
        ts.prev_bytes = 0;

        next_heavy_token = 0;

        uint32_t total_steps = prompt_len + req_max_tokens;
        if (prompt_len > 0) total_steps -= 1;

        // ── Layer-major batched prefill ───────────────────────
        // All prompt tokens except the last go through the chunked
        // layer-major path: per layer, the chunk's routed experts are
        // deduplicated and each blob is claimed once for all tokens
        // that use it. The last prompt token runs through the normal
        // serial step below so sampling/emission stay in one place.
        uint32_t n_start = 0;
        if (prompt_len > 1) {
            static float prefill_attn_scores[ATTN_CTX];
            smoe::prefill::Params pparams {
                .cfg = &cfg,
                .scout = &scout,
                .streamer = &streamer,
                .metal = metal,
                .hdr = &vault_hdr,
                .layout = &expert_layout,
                .full_kv_cache = full_kv_cache,
                .attn_ctx = ATTN_CTX,
                .moe_start_layer = moe_start_layer,
                .rope_inv_freq = rope_inv_freq,
                .b = pf,
                .expert_hits = g_expert_hits,
                .debug = g_debug,
            };
            smoe::prefill::run_layer_major(pparams, prompt_tokens, prompt_len - 1,
                                           stream_base, ctx_pos, ctx_fill,
                                           prefill_attn_scores);
            n_start = prompt_len - 1;
        }

    for (uint32_t n = n_start; n < total_steps; ++n) {
        uint32_t heavy_cur_token;
        bool is_prompt = (n < prompt_len);

        instr_gen = g_instr && !is_prompt;
        double   it_tok0  = 0.0;
        uint64_t it_bytes0 = 0;
        if (instr_gen) {
            it_tok0   = wall_ms();
            it_mark   = it_tok0;
            it_bytes0 = streamer.bytes_read();
        }

        if (is_prompt) {
            heavy_cur_token = prompt_tokens[n];
        } else {
            heavy_cur_token = next_heavy_token;
            // The stream must contain exactly the tokens whose KV exists,
            // so a generated token joins it when it is PROCESSED (its KV
            // is written this step), not when it is sampled. The final
            // sampled token of a request (EOS or cap-end) is never
            // processed; appending it at sampling time left a stream
            // position with no KV entry, so every later serve-mode turn
            // wrote slot = position − 1 while RoPE used the true position,
            // shifting attention one token per turn boundary.
            if (stream_len < STREAM_CAP) {
                stream_tokens[stream_len++] = heavy_cur_token;
            }
        }

        const uint16_t* emb = scout.get_embed() + heavy_cur_token * d_model;
        for (uint32_t d = 0; d < d_model; ++d) {
            heavy_hidden[d] = smoe::bf16_to_f32(emb[d]);
        }

        // Token emission happens once in Step 6; anything here is debug telemetry
        // only and must stay off stdout (chat.py parses stdout as the token stream).
        if (n >= prompt_len && g_debug) {
            std::fprintf(stderr, "[RAW TOK: %u] ", heavy_cur_token);
        }

        if (n == 0) {
            std::fprintf(stderr, "\n");
        }
        if (n == 0 && g_debug) {
            std::fprintf(stderr, "\n[Token %u] emb = %.4f %.4f %.4f %.4f\n", heavy_cur_token, heavy_hidden[0], heavy_hidden[1], heavy_hidden[2], heavy_hidden[3]);
        }

        // No per-token pruning: the ring retains CONSUMED slots as an LRU
        // cache across tokens (and across serve-mode requests). Capacity
        // is reclaimed lazily by prefetch()'s ref-count-guarded LRU
        // eviction, exactly as prefill already does. The old prune ran
        // with an empty lookahead queue at steady state, so it evicted
        // the ENTIRE ring every generated token — zero cross-token reuse
        // despite a measured 53.5% adjacent-token expert overlap.

        // ── Phase B: Heavy Model Execution ─────────────────────
        // The scout's backbone is not run at all — routing is computed
        // exactly from the heavy hidden state at each MoE layer below,
        // and next-token prefetch comes from each layer's own top-16
        // gate ranking (speculative, low-priority queue).
        static float attn_scores[ATTN_CTX];

        const uint16_t* heavy_emb = scout.get_embed() + static_cast<size_t>(heavy_cur_token) * d_model;
        for (uint32_t i = 0; i < d_model; ++i) heavy_hidden[i] = smoe::bf16_to_f32(heavy_emb[i]);

        // RoPE rotation for this ABSOLUTE stream position (not the loop
        // index — in server mode the request starts mid-stream), shared
        // by all heads and layers.
        for (uint32_t d = 0; d < rope_half; ++d) {
            float angle = static_cast<float>(stream_base + n) * rope_inv_freq[d];
            rope_cos[d] = std::cos(angle);
            rope_sin[d] = std::sin(angle);
        }

        for (uint32_t l = 0; l < num_layers; ++l) {
            std::memcpy(heavy_normed, heavy_hidden, d_model * sizeof(float));
            smoe::rms_norm_bf16(heavy_normed, scout.get_input_norm(l), d_model);

            if (metal) {
                const uint16_t* weights[3] = { scout.get_q_proj(l), scout.get_k_proj(l), scout.get_v_proj(l) };
                const float* inputs[3] = { heavy_normed, heavy_normed, heavy_normed };
                float* outputs[3] = { heavy_qbuf, heavy_kbuf, heavy_vbuf };
                uint32_t rows[3] = { q_dim, kv_dim, kv_dim };
                uint32_t cols[3] = { d_model, d_model, d_model };
                smoe_metal_scout_matvec_batch_bf16(metal, weights, inputs, outputs, rows, cols, 3);
            } else {
                smoe::matvec_bf16(heavy_qbuf, scout.get_q_proj(l), heavy_normed, q_dim, d_model);
                smoe::matvec_bf16(heavy_kbuf, scout.get_k_proj(l), heavy_normed, kv_dim, d_model);
                smoe::matvec_bf16(heavy_vbuf, scout.get_v_proj(l), heavy_normed, kv_dim, d_model);
            }

            // Apply per-head Q/K norm
            const uint16_t* q_norm_w = scout.get_q_norm(l);
            const uint16_t* k_norm_w = scout.get_k_norm(l);
            if (q_norm_w) {
                for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                    smoe::rms_norm_bf16(heavy_qbuf + h * cfg.head_dim, q_norm_w, cfg.head_dim);
                }
            }
            if (k_norm_w) {
                for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                    smoe::rms_norm_bf16(heavy_kbuf + h * cfg.head_dim, k_norm_w, cfg.head_dim);
                }
            }

            // Apply RoPE to Q
            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                for (uint32_t d = 0; d < rope_half; ++d) {
                    float q0 = heavy_qbuf[h * cfg.head_dim + d];
                    float q1 = heavy_qbuf[h * cfg.head_dim + d + rope_half];
                    heavy_qbuf[h * cfg.head_dim + d]             = q0 * rope_cos[d] - q1 * rope_sin[d];
                    heavy_qbuf[h * cfg.head_dim + d + rope_half] = q0 * rope_sin[d] + q1 * rope_cos[d];
                }
            }

            // Apply RoPE to K
            for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                for (uint32_t d = 0; d < rope_half; ++d) {
                    float k0 = heavy_kbuf[h * cfg.head_dim + d];
                    float k1 = heavy_kbuf[h * cfg.head_dim + d + rope_half];
                    heavy_kbuf[h * cfg.head_dim + d]             = k0 * rope_cos[d] - k1 * rope_sin[d];
                    heavy_kbuf[h * cfg.head_dim + d + rope_half] = k0 * rope_sin[d] + k1 * rope_cos[d];
                }
            }

            float* k_cache = full_kv_cache + (l * 2 * ATTN_CTX + 0 * ATTN_CTX) * kv_dim;
            float* v_cache = full_kv_cache + (l * 2 * ATTN_CTX + 1 * ATTN_CTX) * kv_dim;
            
            const uint32_t slot = ctx_pos % ATTN_CTX;
            std::memcpy(k_cache + slot * kv_dim, heavy_kbuf, kv_dim * sizeof(float));
            std::memcpy(v_cache + slot * kv_dim, heavy_vbuf, kv_dim * sizeof(float));
            scout.write_kv_cache(l, slot, heavy_kbuf, heavy_vbuf);

            // Multi-Head Attention CPU (with GQA support)
            const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
            const uint32_t valid = (ctx_fill < ATTN_CTX) ? ctx_fill + 1 : ATTN_CTX;
            std::memset(heavy_attn_out, 0, q_dim * sizeof(float));
            
            uint32_t heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

            for (uint32_t h = 0; h < cfg.num_heads; ++h) {
                uint32_t kv_h = h / heads_per_kv;
                for (uint32_t i = 0; i < valid; ++i) {
                    uint32_t ki = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                    float dot_qk = 0.0f;
                    const float* krow = k_cache + ki * kv_dim + kv_h * cfg.head_dim;
                    const float* qhead = heavy_qbuf + h * cfg.head_dim;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        dot_qk += qhead[d] * krow[d];
                    }
                    attn_scores[i] = dot_qk * scale;
                }

                float max_val = attn_scores[0];
                for (uint32_t i = 1; i < valid; ++i) {
                    if (attn_scores[i] > max_val) max_val = attn_scores[i];
                }
                float sum = 0.0f;
                for (uint32_t i = 0; i < valid; ++i) {
                    attn_scores[i] = std::exp(attn_scores[i] - max_val);
                    sum += attn_scores[i];
                }
                float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                for (uint32_t i = 0; i < valid; ++i) {
                    attn_scores[i] *= inv_sum;
                }

                for (uint32_t i = 0; i < valid; ++i) {
                    uint32_t vi = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                    const float alpha = attn_scores[i];
                    const float* vrow = v_cache + vi * kv_dim + kv_h * cfg.head_dim;
                    float* out_head = heavy_attn_out + h * cfg.head_dim;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        out_head[d] += alpha * vrow[d];
                    }
                }
            }

            // O Proj and Residual
            if (metal) {
                smoe_metal_scout_matvec_bf16(metal, scout.get_o_proj(l), heavy_attn_out, heavy_normed, d_model, q_dim);
            } else {
                smoe::matvec_bf16(heavy_normed, scout.get_o_proj(l), heavy_attn_out, d_model, q_dim);
            }
            for (uint32_t i = 0; i < d_model; ++i) {
                heavy_hidden[i] += heavy_normed[i];
            }

            // FFN Norm
            std::memcpy(heavy_normed, heavy_hidden, d_model * sizeof(float));
            smoe::rms_norm_bf16(heavy_normed, scout.get_post_norm(l), d_model);

            // FFN
            if (l == 0 && cfg.has_dense_layer_0) {
                if (metal) {
                    const uint16_t* weights[2] = { scout.get_l0_gate(), scout.get_l0_up() };
                    const float* inputs[2] = { heavy_normed, heavy_normed };
                    float* outputs[2] = { l0_gate_out, l0_up_out };
                    uint32_t rows[2] = { ffn_dim, ffn_dim };
                    uint32_t cols[2] = { d_model, d_model };
                    smoe_metal_scout_matvec_batch_bf16(metal, weights, inputs, outputs, rows, cols, 2);
                } else {
                    smoe::matvec_bf16(l0_gate_out, scout.get_l0_gate(), heavy_normed, ffn_dim, d_model);
                    smoe::matvec_bf16(l0_up_out, scout.get_l0_up(), heavy_normed, ffn_dim, d_model);
                }
                for (uint32_t i = 0; i < ffn_dim; ++i) {
                    // Silu
                    float val = l0_gate_out[i];
                    l0_gate_out[i] = (val / (1.0f + std::exp(-val))) * l0_up_out[i];
                }
                if (metal) {
                    smoe_metal_scout_matvec_bf16(metal, scout.get_l0_down(), l0_gate_out, heavy_normed, d_model, ffn_dim);
                } else {
                    smoe::matvec_bf16(heavy_normed, scout.get_l0_down(), l0_gate_out, d_model, ffn_dim);
                }
                for (uint32_t d = 0; d < d_model; ++d) heavy_hidden[d] += heavy_normed[d];
                if (n == 0 && g_debug) {
                    std::fprintf(stderr, "\n[L0] hidden = %.4f %.4f %.4f %.4f\n", heavy_hidden[0], heavy_hidden[1], heavy_hidden[2], heavy_hidden[3]);
                }
            } else {
                // Exact routing, ALL steps: the real router gate evaluated
                // on the heavy hidden state. One ranked top-(8+spec)
                // selection: ranks 0–7 are the true routing for THIS token
                // (weights renormalised over the 8 per norm_topk_prob —
                // identical to a direct top-8 selection); ranks beyond 8
                // are the speculative prefetch bet for the NEXT token at
                // this layer (measured coverage of the next token's top-8:
                // 46.4% retention alone → 63.7% at +8 → 71.9% at +16).
                // Speculative reads use the streamer's low-priority queue
                // so they can never delay the demand fetches this token is
                // spinning on.
                smoe::matvec_bf16(exact_gate_scores,
                                  scout.get_gate(l - moe_start_layer),
                                  heavy_normed,
                                  cfg.max_experts_per_layer, d_model);
                const uint32_t nsel = scout.compute_top_k(
                    exact_gate_scores, cfg.max_experts_per_layer, 8 + spec_width,
                    exact_rank_ids, exact_rank_w, false);
                const uint32_t n8 = (nsel < 8) ? nsel : 8;
                exact_pred.layer_id = l;
                exact_pred.count = n8;
                float w_sum = 0.0f;
                for (uint32_t e = 0; e < n8; ++e) w_sum += exact_rank_w[e];
                const float w_inv =
                    (cfg.norm_topk_prob && w_sum > 0.0f) ? 1.0f / w_sum : 1.0f;
                for (uint32_t e = 0; e < n8; ++e) {
                    exact_pred.expert_ids[e]     = exact_rank_ids[e];
                    exact_pred.expert_weights[e] = exact_rank_w[e] * w_inv;
                    (void)streamer.prefetch(l, exact_rank_ids[e]);
                }
                for (uint32_t e = n8; e < nsel; ++e) {
                    (void)streamer.prefetch(l, exact_rank_ids[e], /*speculative=*/true);
                }
                const smoe::scout::ExpertPrediction& pred = exact_pred;

                // Rank coverage: would prefetching the PREVIOUS token's
                // top-M gate ranking have covered this token's true top-8?
                // Decides the width of a scout-free speculative prefetch.
                if (g_instr && l < 128) {
                    if (!is_prompt) {
                        for (uint32_t e = 0; e < pred.count; ++e) {
                            for (uint32_t r = 0; r < g_instr_prev_top32_n[l]; ++r) {
                                if (g_instr_prev_top32[l][r] == pred.expert_ids[e]) {
                                    if (r < 8)  instr.cov[0]++;
                                    if (r < 16) instr.cov[1]++;
                                    if (r < 24) instr.cov[2]++;
                                    instr.cov[3]++;   // r < 32 by construction
                                    break;
                                }
                            }
                        }
                        instr.cov_total += pred.count;
                    }
                    static uint32_t t32_ids[32];
                    static float    t32_w[32];
                    uint32_t n32 = scout.compute_top_k(
                        exact_gate_scores, cfg.max_experts_per_layer, 32,
                        t32_ids, t32_w, false);
                    if (n32 > 32) n32 = 32;
                    g_instr_prev_top32_n[l] = static_cast<uint8_t>(n32);
                    std::memcpy(g_instr_prev_top32[l], t32_ids, n32 * sizeof(uint32_t));
                }

                // ── Phase 1: Dispatch Currently Available Routed Experts to GPU ──
                instr_bucket(instr.dense_ms);   // norms+QKV+attention+o_proj+gate
                std::memset(routed_out, 0, d_model * sizeof(float));

                // Adjacent-token routed-expert overlap: how many of this
                // token's experts were also routed by the previous token —
                // the upper bound on what cross-token ring retention buys.
                // Prompt steps seed the masks without counting.
                if (g_instr && l < 128) {
                    uint64_t cur0 = 0, cur1 = 0;
                    for (uint32_t e = 0; e < pred.count; ++e) {
                        const uint32_t id = pred.expert_ids[e];
                        if (id < 64)        cur0 |= 1ULL << id;
                        else if (id < 128)  cur1 |= 1ULL << (id - 64);
                    }
                    if (!is_prompt) {
                        instr.ovl_hits  += __builtin_popcountll(cur0 & g_instr_prev_mask[l][0])
                                         + __builtin_popcountll(cur1 & g_instr_prev_mask[l][1]);
                        instr.ovl_total += pred.count;
                    }
                    g_instr_prev_mask[l][0] = cur0;
                    g_instr_prev_mask[l][1] = cur1;
                }
                bool executed[16] = {false};
                uint32_t num_executed = 0;
                uint64_t spin = 0;
                
                void* wait_handles[16] = {nullptr};
                smoe::io::RingSlot* active_slots[16] = {nullptr};
                static std::vector<float*> expert_hidden_scratch(16, nullptr);
                static std::vector<float*> expert_output_vec(16, nullptr);
                for (int i = 0; i < 16; ++i) {
                    if (!expert_hidden_scratch[i]) expert_hidden_scratch[i] = smoe::allocate_aligned_float(ffn_dim);
                    if (!expert_output_vec[i]) expert_output_vec[i] = smoe::allocate_aligned_float(d_model);
                }
                // Initial non-blocking pass
                if (g_debug) {
                    std::fprintf(stderr, "[DEBUG] layer %u: pred.count = %u. Experts: ", l, pred.count);
                    for (uint32_t e = 0; e < pred.count; ++e) {
                        std::fprintf(stderr, "%u ", pred.expert_ids[e]);
                    }
                    std::fprintf(stderr, "\n");
                }
                std::fflush(stderr);
                
                for (uint32_t e = 0; e < pred.count; ++e) {
                    smoe::io::RingSlot* slot = streamer.claim_specific(pred.layer_id, pred.expert_ids[e]);
                    if (slot) {
                        active_slots[e] = slot;
                        if (pred.layer_id < 128 && pred.expert_ids[e] < smoe::prefill::HITS_STRIDE)
                            g_expert_hits[pred.layer_id * smoe::prefill::HITS_STRIDE + pred.expert_ids[e]]++;
                        if (slot->data_size > 0) {
                            const uint8_t*  packed_gate = slot->data + expert_layout.gate_packed_offset;
                            const uint16_t* scales_gate = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.gate_scales_offset);
                            const uint8_t*  packed_up   = slot->data + expert_layout.up_packed_offset;
                            const uint16_t* scales_up   = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.up_scales_offset);
                            const uint8_t*  packed_down = slot->data + expert_layout.down_packed_offset;
                            const uint16_t* scales_down = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.down_scales_offset);

                            if (g_debug) {
                                std::fprintf(stderr, "[DEBUG] e=%u, ffn_dim=%u, d_model=%u, scratch=%p, out=%p\n", e, ffn_dim, d_model, expert_hidden_scratch[e], expert_output_vec[e]);
                                std::fflush(stderr);
                            }
                            std::memset(expert_hidden_scratch[e], 0, ffn_dim * sizeof(float));
                            std::memset(expert_output_vec[e], 0, d_model * sizeof(float));

                            wait_handles[e] = smoe_metal_fused_ffn(
                                metal,
                                packed_gate, scales_gate, packed_up, scales_up, packed_down, scales_down,
                                heavy_normed, expert_hidden_scratch[e], expert_output_vec[e],
                                expert_layout.gate_rows, expert_layout.gate_cols,
                                expert_layout.down_rows, expert_layout.down_cols,
                                vault_hdr.group_size, vault_hdr.bits, e
                            );
                        }
                        executed[e] = true;
                        num_executed++;
                    }
                }
                instr_bucket(instr.dispatch_ms);

                // ── Phase 2: Compute Shared Expert on CPU while GPU/SSD works ──
                std::memset(shared_out, 0, d_model * sizeof(float));
                
                if (scout.get_shared_gate(l)) {
                    if (metal) {
                        const uint16_t* weights[2] = { scout.get_shared_gate(l), scout.get_shared_up(l) };
                        const float* inputs[2] = { heavy_normed, heavy_normed };
                        float* outputs[2] = { shared_gate_out, shared_up_out };
                        uint32_t rows[2] = { shared_dim, shared_dim };
                        uint32_t cols[2] = { d_model, d_model };
                        smoe_metal_scout_matvec_batch_bf16(metal, weights, inputs, outputs, rows, cols, 2);
                    } else {
                        smoe::matvec_bf16(shared_gate_out, scout.get_shared_gate(l), heavy_normed, shared_dim, d_model);
                        smoe::matvec_bf16(shared_up_out, scout.get_shared_up(l), heavy_normed, shared_dim, d_model);
                    }
                    for (uint32_t i = 0; i < shared_dim; ++i) {
                        float val = shared_gate_out[i];
                        shared_gate_out[i] = (val / (1.0f + std::exp(-val))) * shared_up_out[i];
                    }
                    if (metal) {
                        smoe_metal_scout_matvec_bf16(metal, scout.get_shared_down(l), shared_gate_out, shared_out, d_model, shared_dim);
                    } else {
                        smoe::matvec_bf16(shared_out, scout.get_shared_down(l), shared_gate_out, d_model, shared_dim);
                    }
                }

                instr_bucket(instr.dense_ms);   // shared expert (none on Qwen3)

                // ── Phase 3: Spin-wait for any remaining missing experts ──
                while (num_executed < pred.count) {
                    bool made_progress = false;
                    for (uint32_t e = 0; e < pred.count; ++e) {
                        if (executed[e]) continue;

                        smoe::io::RingSlot* slot = streamer.claim_specific(pred.layer_id, pred.expert_ids[e]);
                        if (slot) {
                            active_slots[e] = slot;
                            if (pred.layer_id < 128 && pred.expert_ids[e] < smoe::prefill::HITS_STRIDE)
                                g_expert_hits[pred.layer_id * smoe::prefill::HITS_STRIDE + pred.expert_ids[e]]++;
                            if (slot->data_size > 0) {
                                const uint8_t*  packed_gate = slot->data + expert_layout.gate_packed_offset;
                                const uint16_t* scales_gate = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.gate_scales_offset);
                                const uint8_t*  packed_up   = slot->data + expert_layout.up_packed_offset;
                                const uint16_t* scales_up   = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.up_scales_offset);
                                const uint8_t*  packed_down = slot->data + expert_layout.down_packed_offset;
                                const uint16_t* scales_down = reinterpret_cast<const uint16_t*>(slot->data + expert_layout.down_scales_offset);

                                std::memset(expert_hidden_scratch[e], 0, ffn_dim * sizeof(float));
                                std::memset(expert_output_vec[e], 0, d_model * sizeof(float));

                                wait_handles[e] = smoe_metal_fused_ffn(
                                    metal,
                                    packed_gate, scales_gate, packed_up, scales_up, packed_down, scales_down,
                                    heavy_normed, expert_hidden_scratch[e], expert_output_vec[e],
                                    expert_layout.gate_rows, expert_layout.gate_cols,
                                    expert_layout.down_rows, expert_layout.down_cols,
                                    vault_hdr.group_size, vault_hdr.bits, e
                                );
                            }
                            executed[e] = true;
                            num_executed++;
                            if (g_debug) {
                                float test_sum = 0;
                                for (int i=0; i<10; ++i) test_sum += expert_output_vec[e][i];
                                std::fprintf(stderr, "[DEBUG] layer=%u e=%d weight=%f sum_10=%f\n", l, e, pred.expert_weights[e], test_sum);
                            }
                            made_progress = true;
                        } else {
                            
if (spin % 10000 == 0) {

                                streamer.prefetch(pred.layer_id, pred.expert_ids[e]);
                            }
                        }
                    }

                    if (!made_progress) {
                        std::this_thread::yield();
                        spin++;
                    }
                }
                instr_bucket(instr.io_spin_ms);

                // ── Phase 4: Wait for GPU and accumulate ──
                for (uint32_t e = 0; e < pred.count; ++e) {
                    if (wait_handles[e]) {
                        smoe_metal_wait(metal, wait_handles[e], expert_output_vec[e], e, d_model);
                        float weight = pred.expert_weights[e];
                        if (l == 1 && n == 0 && g_debug) {
                            std::fprintf(stderr, "[L1] Routed Expert %d, weight = %.4f\n", pred.expert_ids[e], weight);
                        }
                        for (uint32_t d = 0; d < d_model; ++d) {
                            if (g_debug && d == 0) std::fprintf(stderr, "[MoE_OUT] layer=%u e=%d sum_first=%f\n", l, e, expert_output_vec[e][0]);
                            routed_out[d] += weight * expert_output_vec[e][d];
                        }
                    }
                    if (active_slots[e]) {
                        streamer.release(active_slots[e]);
                    }
                }
                smoe_metal_swap_buffers(metal);
                instr_bucket(instr.gpu_wait_ms);

                if (l == 1 && n == 0 && g_debug) {
                    std::fprintf(stderr, "\n[L1] shared_out = %.4f %.4f %.4f %.4f\n", shared_out[0], shared_out[1], shared_out[2], shared_out[3]);
                    std::fprintf(stderr, "[L1] routed_out = %.4f %.4f %.4f %.4f\n", routed_out[0], routed_out[1], routed_out[2], routed_out[3]);
                }

                // Add FFN residuals
                for (uint32_t d = 0; d < d_model; ++d) {
                    heavy_hidden[d] += shared_out[d] + routed_out[d];
                }

                if (l == 1 && n == 0 && g_debug) {
                    std::fprintf(stderr, "\n[L1] hidden after FFN = %.4f %.4f %.4f %.4f\n", heavy_hidden[0], heavy_hidden[1], heavy_hidden[2], heavy_hidden[3]);
                }
                // (update_echo removed: heavy_echo was written every layer
                // and read by nothing — dead machinery from the abandoned
                // temporal-routing-feedback design.)
            }
        }
        
        ctx_pos = (ctx_pos + 1) % ATTN_CTX;
        if (ctx_fill < ATTN_CTX) ++ctx_fill;

        // ── Step 5: Final Model Norm and LM Head ──────────────
        // Prompt positions before the last already know their next token,
        // so the vocab-sized LM head matvec (~622M MACs on CPU) and the
        // sampling pass would be discarded work — skip them entirely.
        // heavy_hidden is rebuilt from the embedding at the top of every
        // step, so leaving it un-normed here is safe.
        bool is_generating = (prompt_len == 0) || (n >= prompt_len - 1);
        if (is_generating) {
            if (g_debug) {
                float h_sum2 = 0.0f;
                for(uint32_t i=0; i<d_model; i++) h_sum2 += heavy_hidden[i]*heavy_hidden[i];
                std::fprintf(stderr, "\n[n=%u] raw_hidden_L2 = %f\n", n, std::sqrt(h_sum2));
            }
            smoe::rms_norm_bf16(heavy_hidden, scout.get_model_norm(), d_model);

            // LM Head: 151936x4096 matvec — GPU when available (the scout
            // already registered w_lm_head and lm_head_scores with Metal;
            // heavy_hidden is registered at init above).
            float* scores = scout.get_lm_head_scores();
            if (metal) {
                smoe_metal_scout_matvec_bf16(metal, scout.get_lm_head(), heavy_hidden, scores, cfg.vocab_size, d_model);
            } else {
                smoe::matvec_bf16(scores, scout.get_lm_head(), heavy_hidden, cfg.vocab_size, d_model);
            }

            // Sample Token
            smoe::SamplerConfig sampler_cfg {
                .vocab_size = cfg.vocab_size,
                .temperature = temperature,
                .top_p = top_p,
                .top_k = top_k,
                .rep_penalty = rep_penalty
            };
            uint32_t best_tok = smoe::sample_token(scores, sampler_cfg, stream_tokens, stream_len, rng, sampler_scratch);

            heavy_cur_token = best_tok;
            if (g_debug) {
                float h_sum2 = 0.0f;
                for(uint32_t i=0; i<d_model; i++) h_sum2 += heavy_hidden[i]*heavy_hidden[i];
                std::fprintf(stderr, "\\n[n=%u] hidden_L2 = %f, best_tok = %u\\n", n, std::sqrt(h_sum2), best_tok);
                std::fflush(stderr);
            }
            next_heavy_token = best_tok; // Save for the next contiguous iteration
            // NOTE: best_tok is NOT appended to stream_tokens here — it
            // enters the stream at the top of the step that processes it.
        }
        instr_bucket(instr.lm_ms);

        // Token accounting happens before the EOS break in Step 6 so the
        // final generated token's timing is not lost.
        if (instr_gen) {
            instr.total_ms += wall_ms() - it_tok0;
            instr.bytes    += streamer.bytes_read() - it_bytes0;
            instr.tokens   += 1;
        }

        // ── Step 6: Emit token ────────────────────────────────
        if (is_generating) {
            if (std::find(eos_ids.begin(), eos_ids.end(), heavy_cur_token) != eos_ids.end()) {
                if (g_debug) std::fprintf(stderr, "\n[EOS reached]\n");
                break;
            }
            if (g_raw_ids) {
                std::printf("%u ", heavy_cur_token);
            } else if (g_vocab_loaded && heavy_cur_token < cfg.vocab_size) {
                std::string s = g_vocab[heavy_cur_token];
                size_t pos;
                while ((pos = s.find("Ġ")) != std::string::npos) s.replace(pos, 2, " ");
                while ((pos = s.find("Ċ")) != std::string::npos) s.replace(pos, 2, "\n");
                std::fputs(s.c_str(), stdout);
            } else {
                unsigned char c = static_cast<unsigned char>(heavy_cur_token & 0xFF);
                std::fputc(c, stdout);
            }
            std::fflush(stdout);
        }

        // ── Step 7: Telemetry ─────────────────────────────────
        if (g_debug && is_generating && ((n + 1) % TELEMETRY_EVERY == 0)) {
            print_telemetry(n + 1 - (prompt_len > 0 ? prompt_len - 1 : 0), streamer, ts);
        }
    }

        // ── Request epilogue ──────────────────────────────────
        if (g_instr && instr.tokens > 0) {
            const double nt = double(instr.tokens);
            const double accounted = instr.prune_ms + instr.dense_ms +
                                     instr.dispatch_ms + instr.io_spin_ms +
                                     instr.gpu_wait_ms + instr.lm_ms;
            const double other_ms = instr.total_ms - accounted;
            auto pct = [&](double v) {
                return instr.total_ms > 0.0 ? 100.0 * v / instr.total_ms : 0.0;
            };
            std::fprintf(stderr,
                "\n[instr] %llu generated tokens, %.0f ms/token (%.2f t/s)\n"
                "[instr]   dense    %8.1f ms/tok  %5.1f%%\n"
                "[instr]   dispatch %8.1f ms/tok  %5.1f%%\n"
                "[instr]   io-spin  %8.1f ms/tok  %5.1f%%\n"
                "[instr]   gpu-wait %8.1f ms/tok  %5.1f%%\n"
                "[instr]   lm-head  %8.1f ms/tok  %5.1f%%\n"
                "[instr]   prune    %8.1f ms/tok  %5.1f%%\n"
                "[instr]   other    %8.1f ms/tok  %5.1f%%\n"
                "[instr]   NVMe %.2f GB/token, effective %.2f GB/s over decode\n"
                "[instr]   adjacent-token expert overlap: %.1f%%  (%llu / %llu)\n"
                "[instr]   prev-token top-8/16/24/32 coverage of true top-8: %.1f%% / %.1f%% / %.1f%% / %.1f%%\n",
                static_cast<unsigned long long>(instr.tokens),
                instr.total_ms / nt, nt * 1000.0 / instr.total_ms,
                instr.dense_ms / nt,    pct(instr.dense_ms),
                instr.dispatch_ms / nt, pct(instr.dispatch_ms),
                instr.io_spin_ms / nt,  pct(instr.io_spin_ms),
                instr.gpu_wait_ms / nt, pct(instr.gpu_wait_ms),
                instr.lm_ms / nt,       pct(instr.lm_ms),
                instr.prune_ms / nt,    pct(instr.prune_ms),
                other_ms / nt,          pct(other_ms),
                double(instr.bytes) / nt / 1e9,
                instr.total_ms > 0.0 ? double(instr.bytes) / 1e6 / instr.total_ms : 0.0,
                instr.ovl_total ? 100.0 * double(instr.ovl_hits) / double(instr.ovl_total) : 0.0,
                static_cast<unsigned long long>(instr.ovl_hits),
                static_cast<unsigned long long>(instr.ovl_total),
                instr.cov_total ? 100.0 * double(instr.cov[0]) / double(instr.cov_total) : 0.0,
                instr.cov_total ? 100.0 * double(instr.cov[1]) / double(instr.cov_total) : 0.0,
                instr.cov_total ? 100.0 * double(instr.cov[2]) / double(instr.cov_total) : 0.0,
                instr.cov_total ? 100.0 * double(instr.cov[3]) / double(instr.cov_total) : 0.0);
            instr.reset();
        }

        // Resume prewarm with the freshly updated popularity histogram:
        // it re-warms the ring while the user types the next message.
        save_expert_freq("vault/expert_freq.bin");
        engine_busy.store(false, std::memory_order_relaxed);
        prewarm_epoch.fetch_add(1, std::memory_order_relaxed);

        if (serve) {
            // Terminate the token stream for this request. chat.py reads
            // token IDs until it sees the <<DONE>> sentinel line.
            std::printf("\n<<DONE>>\n");
            std::fflush(stdout);
        }
    } // ═══ end REQUEST LOOP ═══

    smoe::free_aligned_float(full_kv_cache);

    // Final telemetry flush
    if (g_debug) {
        print_telemetry(max_tokens, streamer, ts);
        std::fprintf(stderr, "\n\n");

        // Print final stats
        std::fprintf(stderr,
            "  ─────────────────────────────────────────\n"
            "  Tokens generated : %u\n"
            "  Bytes read       : %.2f GB\n"
            "  Kernel dispatches: %llu\n"
            "  ─────────────────────────────────────────\n\n",
            max_tokens,
            double(streamer.bytes_read()) / 1e9,
            static_cast<unsigned long long>(smoe_metal_kernel_dispatches(metal)));
    }

    // ── Shutdown ──────────────────────────────────────────────
    prewarm_shutdown.store(true, std::memory_order_relaxed);
    if (prewarm_thread.joinable()) prewarm_thread.join();
    save_expert_freq("vault/expert_freq.bin");
    streamer.shutdown();
    smoe_metal_destroy(metal);

    return 0;
}
