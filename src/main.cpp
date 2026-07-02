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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach.h>
#include <thread>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <vector>

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

struct ExpertLayout {
    uint32_t gate_rows;
    uint32_t gate_cols;
    uint64_t gate_packed_offset;
    uint64_t gate_scales_offset;

    uint32_t up_rows;
    uint32_t up_cols;
    uint64_t up_packed_offset;
    uint64_t up_scales_offset;

    uint32_t down_rows;
    uint32_t down_cols;
    uint64_t down_packed_offset;
    uint64_t down_scales_offset;
};

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
    // ring_size: target ≤ 25% of available physical RAM for the ring pool
    //   so the OS is never pressured into OOM-killing us. On a 16 GB machine
    //   this yields ~3 GB for the ring; on a 128 GB machine, ~24 GB.
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

        // ── 2. Ring size: use 25% of free physical RAM ─────────
        if (ring_size == 0) {
            // Query available RAM via host_statistics64
            uint64_t free_bytes = 0;
            vm_statistics64_data_t vmstat {};
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                                  reinterpret_cast<host_info64_t>(&vmstat),
                                  &count) == KERN_SUCCESS) {
                // free_count includes pages that are available without swapping
                uint64_t page_sz = static_cast<uint64_t>(vm_page_size);
                free_bytes = (static_cast<uint64_t>(vmstat.free_count) +
                              static_cast<uint64_t>(vmstat.inactive_count)) * page_sz;
            }
            if (free_bytes < 512ULL * 1024 * 1024)
                free_bytes = 512ULL * 1024 * 1024; // assume at least 512 MB

            uint64_t slot_bytes_now = slot_mb * 1024ULL * 1024ULL;
            uint64_t ring_budget    = free_bytes / 4; // 25% of free RAM
            uint32_t auto_ring      = static_cast<uint32_t>(ring_budget / slot_bytes_now);

            // Clamp: minimum 64 (one full token's experts + spare), maximum 4096
            if (auto_ring < 64)   auto_ring = 64;
            if (auto_ring > 4096) auto_ring = 4096;
            ring_size = auto_ring;
            std::fprintf(stderr, "[ring] Auto ring size : %u slots × %llu MB = %.1f GB  (free RAM: %.1f GB)\n",
                ring_size, (unsigned long long)slot_mb,
                static_cast<double>(ring_size) * slot_mb / 1024.0,
                static_cast<double>(free_bytes) / (1024.0*1024*1024));
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

    // ── Cache Pre-Warming (Stealth Background Load) ───────────
    std::thread prewarm_thread([&streamer, vault_hdr, &cfg]() {
        uint32_t count = 0;
        // Leave 800 slots empty to ensure there's room for all active experts (8 per layer * 94 layers = ~752)
        uint32_t max_prewarm = (streamer.ring_size() > 800) ? streamer.ring_size() - 800 : 0;
        for (uint32_t l = 0; l < cfg.num_moe_layers && count < max_prewarm; ++l) {
            for (uint32_t e = 0; e < vault_hdr.max_experts_per_layer && count < max_prewarm; ++e) {
                if (streamer.prefetch(l, e)) count++;
                // Sleep to trickle-load the cache, keeping the MPMC queue empty 
                // and the NVMe SSD available for immediate priority prompt prefetching!
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
        }
    });
    prewarm_thread.detach();

    const uint32_t d_model = cfg.d_model;
    const uint32_t ffn_dim = cfg.ffn_dim;
    const uint32_t shared_dim = cfg.shared_expert_ffn_dim; // 0 = no shared expert (Qwen3-235B), >0 = DeepSeek
    const uint32_t moe_start_layer = cfg.has_dense_layer_0 ? 1 : 0;


    // ── Load vocabulary ───────────────────────────────────────
    load_vocab("vault/vocab.bin", cfg.vocab_size);

    // ── Tokenise prompt ───────────────────────────────────────
    // Pre-allocate on the stack — no heap in the loop.
    uint32_t* prompt_tokens = (uint32_t*)__builtin_alloca(4096 * sizeof(uint32_t));
    uint32_t        prompt_len = 0;
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

    // ── Pre-generate telemetry state ──────────────────────────
    TelemetryState ts;
    ts.start_ms   = wall_ms();
    ts.prev_ms    = ts.start_ms;
    ts.prev_bytes = 0;

    // ── Sampling state ────────────────────────────────────────
    static uint32_t token_history[8192];
    uint32_t history_len = 0;
    for (uint32_t i = 0; i < prompt_len; ++i) {
        if (history_len < 8192) {
            token_history[history_len++] = prompt_tokens[i];
        }
    }
    std::mt19937 rng(1337); // Fixed seed for deterministic debug, can be randomized later

    // ── Print prompt passthrough ──────────────────────────────
    if (g_debug) {
        std::fprintf(stdout, "\n");
        for (uint32_t i = 0; i < prompt_len; ++i) {
            uint32_t tok = prompt_tokens[i];
            if (g_vocab_loaded && tok < cfg.vocab_size) {
                std::string s = g_vocab[tok];
                size_t pos;
                while ((pos = s.find("Ġ")) != std::string::npos) s.replace(pos, 2, " ");
                while ((pos = s.find("Ċ")) != std::string::npos) s.replace(pos, 2, "\n");
                std::fputs(s.c_str(), stdout);
            } else {
                unsigned char c = static_cast<unsigned char>(tok & 0xFF);
                std::fputc(c, stdout);
            }
        }
    }

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

    uint32_t total_steps = prompt_len + max_tokens;
    if (prompt_len > 0) total_steps -= 1;

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

    smoe_metal_register_buffer(metal, heavy_hidden, heavy_hidden_aligned);
    smoe_metal_register_buffer(metal, heavy_normed, heavy_normed_aligned);
    smoe_metal_register_buffer(metal, heavy_qbuf,   qbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_kbuf,   kvbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_vbuf,   kvbuf_aligned);
    smoe_metal_register_buffer(metal, heavy_attn_out, heavy_attn_out_aligned);

    uint32_t heavy_cur_token = (prompt_len > 0) ? prompt_tokens[0] : 0;
    uint32_t scout_cur_token = heavy_cur_token;

    // Speculative lookahead queue
    static constexpr uint32_t GEN_LOOKAHEAD_K = 1;
    static constexpr uint32_t MAX_LOOKAHEAD = 64;
    static smoe::scout::ScoutOutput scout_queue[MAX_LOOKAHEAD];
    uint32_t sq_head = 0;
    uint32_t sq_tail = 0;
    uint32_t sq_size = 0;

    struct ActiveExpertsContext {
        const smoe::scout::ScoutOutput* queue;
        uint32_t head;
        uint32_t size;
        uint32_t num_layers;
        uint32_t moe_start_layer;
    };

    auto is_expert_active = [](uint32_t layer_id, uint32_t expert_id, void* ctx) -> bool {
        auto* c = static_cast<ActiveExpertsContext*>(ctx);
        for (uint32_t i = 0; i < c->size; ++i) {
            uint32_t idx = (c->head + i) % MAX_LOOKAHEAD;
            const auto& scout_out = c->queue[idx];
            const uint32_t moe_start = c->moe_start_layer;
            if (layer_id < moe_start) continue;
            const uint32_t routing_idx = layer_id - moe_start;
            if (routing_idx < c->num_layers) {
                const auto& pred = scout_out.routing[routing_idx];
                for (uint32_t e = 0; e < pred.count; ++e) {
                    if (pred.expert_ids[e] == expert_id) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    uint32_t next_heavy_token = 0;

    for (uint32_t n = 0; n < total_steps; ++n) {
        uint32_t heavy_cur_token;
        bool is_prompt = (n < prompt_len);

        if (is_prompt) {
            heavy_cur_token = prompt_tokens[n];
            scout_cur_token = prompt_tokens[n];
        } else {
            heavy_cur_token = next_heavy_token;
        }

        if (is_prompt) {
            // During prompt, we do not lookahead. We only evaluate the exact token.
            sq_size = 0;
            sq_head = 0;
            sq_tail = 0;
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

        // Clean up any leaked/discarded slots in the streamer
        {
            ActiveExpertsContext act_ctx { scout_queue, sq_head, sq_size, cfg.num_moe_layers, moe_start_layer };
            streamer.prune_slots(is_expert_active, &act_ctx);
        }

        // If the lookahead queue is empty (due to prompt, divergence, or heavy catching up),
        // the Scout MUST sync with the Heavy model's current token to provide accurate routing.
        if (sq_size == 0) {
            scout_cur_token = heavy_cur_token;
        }

        // ── Phase A: Scout Speculative Lookahead ──────────────
        // During prompt: prefetch exactly 1 token's experts at a time (serial, safe).
        // During generation: keep GEN_LOOKAHEAD_K steps ahead for speculative execution.
        // The previous value of MIN(prompt_len-n, 64) during prompt exhausted the ring
        // (64 × ~162 experts >> ring=256 slots) causing a deadlock after the first token.
        uint32_t target_sq_size = is_prompt ? 1 : GEN_LOOKAHEAD_K;

        while (sq_size < target_sq_size) {
            uint32_t scout_input;
            if (is_prompt && (n + sq_size < prompt_len)) {
                scout_input = prompt_tokens[n + sq_size];
            } else {
                scout_input = scout_cur_token;
            }
            smoe::scout::ScoutOutput sout = scout.forward(scout_input);
            scout_queue[sq_tail] = sout;
            sq_tail = (sq_tail + 1) % MAX_LOOKAHEAD;
            sq_size++;

            // Prefetch the experts predicted by the Scout for this step
            for (uint32_t l = 0; l < cfg.num_moe_layers; ++l) {
                const smoe::scout::ExpertPrediction& pred = sout.routing[l];
                for (uint32_t e = 0; e < pred.count; ++e) {
                    (void)streamer.prefetch(pred.layer_id, pred.expert_ids[e]);
                }
            }
            
            // Advance the scout for the next speculative step
            scout_cur_token = sout.next_token_id;
        }

        // ── Phase B: Heavy Model Execution ─────────────────────
        // Claim the routing map for the current step from the queue head
        smoe::scout::ScoutOutput scout_out = scout_queue[sq_head];
        static float attn_scores[ATTN_CTX];

        const uint16_t* heavy_emb = scout.get_embed() + static_cast<size_t>(heavy_cur_token) * d_model;
        for (uint32_t i = 0; i < d_model; ++i) heavy_hidden[i] = smoe::bf16_to_f32(heavy_emb[i]);

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
                for (uint32_t d = 0; d < cfg.head_dim / 2; ++d) {
                    float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(d * 2) / static_cast<float>(cfg.head_dim));
                    float angle = static_cast<float>(n) * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);
                    
                    float q0 = heavy_qbuf[h * cfg.head_dim + d];
                    float q1 = heavy_qbuf[h * cfg.head_dim + d + (cfg.head_dim / 2)];
                    heavy_qbuf[h * cfg.head_dim + d]                           = q0 * cos_val - q1 * sin_val;
                    heavy_qbuf[h * cfg.head_dim + d + (cfg.head_dim / 2)]      = q0 * sin_val + q1 * cos_val;
                }
            }
            
            // Apply RoPE to K
            for (uint32_t h = 0; h < cfg.num_kv_heads; ++h) {
                for (uint32_t d = 0; d < cfg.head_dim / 2; ++d) {
                    float freq = 1.0f / std::pow(cfg.rope_theta, static_cast<float>(d * 2) / static_cast<float>(cfg.head_dim));
                    float angle = static_cast<float>(n) * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);
                    
                    float k0 = heavy_kbuf[h * cfg.head_dim + d];
                    float k1 = heavy_kbuf[h * cfg.head_dim + d + (cfg.head_dim / 2)];
                    heavy_kbuf[h * cfg.head_dim + d]                           = k0 * cos_val - k1 * sin_val;
                    heavy_kbuf[h * cfg.head_dim + d + (cfg.head_dim / 2)]      = k0 * sin_val + k1 * cos_val;
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
                // ── Phase 1: Dispatch Currently Available Routed Experts to GPU ──
                std::memset(routed_out, 0, d_model * sizeof(float));

                const smoe::scout::ExpertPrediction& pred = scout_out.routing[l - moe_start_layer];
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

                // ── Phase 3: Spin-wait for any remaining missing experts ──
                while (num_executed < pred.count) {
                    bool made_progress = false;
                    for (uint32_t e = 0; e < pred.count; ++e) {
                        if (executed[e]) continue;

                        smoe::io::RingSlot* slot = streamer.claim_specific(pred.layer_id, pred.expert_ids[e]);
                        if (slot) {
                            active_slots[e] = slot;
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
                
                // Temporal routing feedback: pass the heavy state back to the Scout for the next token
                scout.update_echo(l, heavy_hidden);
            }
        }
        
        ctx_pos = (ctx_pos + 1) % ATTN_CTX;
        if (ctx_fill < ATTN_CTX) ++ctx_fill;

        // ── Step 5: Final Model Norm and LM Head ──────────────
        if (g_debug) {
            float h_sum2 = 0.0f;
            for(uint32_t i=0; i<d_model; i++) h_sum2 += heavy_hidden[i]*heavy_hidden[i];
            std::fprintf(stderr, "\n[n=%u] raw_hidden_L2 = %f\n", n, std::sqrt(h_sum2));
        }
        smoe::rms_norm_bf16(heavy_hidden, scout.get_model_norm(), d_model);
        
        // Execute LM Head on CPU
        float* scores = scout.get_lm_head_scores();
        smoe::matvec_bf16(scores, scout.get_lm_head(), heavy_hidden, cfg.vocab_size, d_model);
        uint32_t best_tok   = 0;
        
        // Sample Token
        smoe::SamplerConfig sampler_cfg {
            .vocab_size = cfg.vocab_size,
            .temperature = temperature,
            .top_p = top_p,
            .top_k = top_k,
            .rep_penalty = rep_penalty
        };
        best_tok = smoe::sample_token(scores, sampler_cfg, token_history, history_len, rng);
        
        heavy_cur_token = best_tok;
        if (g_debug) {
            float h_sum2 = 0.0f;
            for(uint32_t i=0; i<d_model; i++) h_sum2 += heavy_hidden[i]*heavy_hidden[i];
            std::fprintf(stderr, "\\n[n=%u] hidden_L2 = %f, best_tok = %u\\n", n, std::sqrt(h_sum2), best_tok);
            std::fflush(stderr);
        }
        next_heavy_token = best_tok; // Save for the next contiguous iteration
        bool is_generating = (prompt_len == 0) || (n >= prompt_len - 1);
        if (is_generating && history_len < 8192) {
            token_history[history_len++] = heavy_cur_token;
        }
        uint32_t expected_scout_token = scout_out.next_token_id;
        if (g_debug) {
            std::fprintf(stderr, "[Scout:%u] ", expected_scout_token);
            std::fflush(stderr);
        }

        // ── Phase C: Speculative Divergence Check ─────────────
        if (!is_prompt && heavy_cur_token != expected_scout_token) {
            // Divergence! Scout guessed wrong.
            // Rollback scout KV-cache by the number of steps it is currently ahead.
            if (sq_size > 1) {
                scout.rollback(sq_size - 1);
            }
            
            // Flush the lookahead queue
            sq_size = 0;
            sq_head = 0;
            sq_tail = 0;
        } else {
            // Scout guessed correctly (or we are in prompt phase)
            // Consume the head of the queue.
            sq_head = (sq_head + 1) % MAX_LOOKAHEAD;
            sq_size--;
        }

        // Always prune discarded experts (LRU cache eviction)
        ActiveExpertsContext act_ctx { scout_queue, sq_head, sq_size, cfg.num_moe_layers, moe_start_layer };
        streamer.prune_slots(is_expert_active, &act_ctx);

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
    streamer.shutdown();
    smoe_metal_destroy(metal);

    return 0;
}
