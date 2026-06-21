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

// ── Compile-time defaults ─────────────────────────────────────
inline constexpr uint32_t DEFAULT_RING_SIZE    = 1024;
inline constexpr uint32_t DEFAULT_WORKERS      = 4;
inline constexpr uint64_t DEFAULT_SLOT_MB      = 8;
inline constexpr uint32_t DEFAULT_MAX_TOKENS   = 512;
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

// ── CLI usage ─────────────────────────────────────────────────

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "\n"
        "  S-MoE Engine — Streaming Mixture-of-Experts Inference\n"
        "  ───────────────────────────────────────────────────────\n"
        "  Usage:\n"
        "    %s --vault <path.smoe> --prompt <text> [options]\n"
        "\n"
        "  Required:\n"
        "    --vault   <path>    Path to the .smoe vault file\n"
        "    --prompt  <text>    Input prompt (string)\n"
        "\n"
        "  Optional:\n"
        "    --scout   <path>    Path to Scout .safetensors weights\n"
        "                        (omit for heuristic-only mode)\n"
        "    --tokens  <N>       Max tokens to generate (default: %u)\n"
        "    --ring    <N>       Ring buffer slot count  (default: %u)\n"
        "    --workers <N>       I/O worker thread count (default: %u)\n"
        "    --slot-mb <N>       Bytes per ring slot, MB (default: %llu)\n"
        "    --raw-ids           Print raw token IDs as integers instead of text\n"
        "\n"
        "  Example:\n"
        "    %s --vault vault/deepseek.smoe --prompt \"Explain MoE routing\" --tokens 200\n"
        "\n",
        argv0,
        DEFAULT_MAX_TOKENS,
        DEFAULT_RING_SIZE,
        DEFAULT_WORKERS,
        DEFAULT_SLOT_MB,
        argv0);
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
static std::string g_vocab[102400];
static bool g_debug = false;
static bool g_raw_ids = false;

static void load_vocab(const char* vocab_bin_path) {
    FILE* f = std::fopen(vocab_bin_path, "rb");
    if (!f) {
        std::fprintf(stderr, "[vocab] ⚠ Failed to open vocabulary file '%s'\n", vocab_bin_path);
        return;
    }
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < 102400; ++i) {
        uint32_t len = 0;
        if (std::fread(&len, sizeof(len), 1, f) != 1) break;
        g_vocab[i].resize(len);
        if (len > 0) {
            if (std::fread(&g_vocab[i][0], 1, len, f) != len) break;
        }
        loaded++;
    }
    std::fclose(f);
    if (loaded == 102400) {
        g_vocab_loaded = true;
        if (g_debug) {
            std::fprintf(stderr, "[vocab] ✓ Loaded 102,400 tokens from '%s'\n", vocab_bin_path);
        }
    } else {
        std::fprintf(stderr, "[vocab] ⚠ Only loaded %u/102,400 tokens\n", loaded);
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

// ── Main entry point ──────────────────────────────────────────

int main(int argc, char* argv[]) {

    // ── Argument parsing ──────────────────────────────────────
    const char* vault_path  = nullptr;
    const char* scout_path  = nullptr;
    const char* prompt_text = nullptr;
    const char* tokens_in   = nullptr;
    uint32_t    max_tokens  = DEFAULT_MAX_TOKENS;
    uint32_t    ring_size   = DEFAULT_RING_SIZE;
    uint32_t    num_workers = DEFAULT_WORKERS;
    uint64_t    slot_mb     = DEFAULT_SLOT_MB;
    float       temperature = 0.6f;
    float       top_p       = 0.9f;
    uint32_t    top_k       = 50;
    float       rep_penalty = 1.0f;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if      (arg("--vault"))     { vault_path  = argv[++i]; }
        else if (arg("--scout"))     { scout_path  = argv[++i]; }
        else if (arg("--prompt"))    { prompt_text = argv[++i]; }
        else if (arg("--tokens-in")) { tokens_in   = argv[++i]; }
        else if (arg("--tokens"))    { max_tokens  = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--ring"))      { ring_size   = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--workers"))   { num_workers = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--slot-mb"))   { slot_mb     = static_cast<uint64_t>(std::atoll(argv[++i])); }
        else if (arg("--temperature")){ temperature = static_cast<float>(std::atof(argv[++i])); }
        else if (arg("--top-p"))      { top_p = static_cast<float>(std::atof(argv[++i])); }
        else if (arg("--top-k"))      { top_k = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--rep-penalty")){ rep_penalty = static_cast<float>(std::atof(argv[++i])); }
        else if (std::strcmp(argv[i], "--debug") == 0) { g_debug = true; }
        else if (std::strcmp(argv[i], "--raw-ids") == 0) { g_raw_ids = true; }
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h")     == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!vault_path || (!prompt_text && !tokens_in)) {
        std::fprintf(stderr,
            "\n  ✗  --vault and either --prompt or --tokens-in are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    const char* sm_dbg = std::getenv("SMOE_DEBUG");
    if (sm_dbg) {
        g_debug = g_debug || (std::strcmp(sm_dbg, "1") == 0 || std::strcmp(sm_dbg, "true") == 0);
    }

    // ── Vault header read ─────────────────────────────────────
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

    // ── Phase 3: Metal bridge init ────────────────────────────
    uint64_t slot_bytes = slot_mb * 1024ULL * 1024ULL;
    SmoeMetalCtx* metal = smoe_metal_init(slot_bytes);
    if (!metal) {
        std::fprintf(stderr, "  ✗  Metal initialisation failed.\n");
        return 1;
    }

    // ── Phase 2: Streamer init ────────────────────────────────
    smoe::io::Streamer streamer(vault_path, ring_size, num_workers, slot_bytes);
    smoe_metal_register_buffer(metal, streamer.pool_data(), streamer.pool_size());

    // ── Cache Pre-Warming (Stealth Background Load) ───────────
    std::thread prewarm_thread([&streamer, vault_hdr]() {
        uint32_t count = 0;
        // Leave 256 slots empty for speculative execution
        uint32_t max_prewarm = (streamer.ring_size() > 256) ? streamer.ring_size() - 256 : 0;
        for (uint32_t l = 0; l < smoe::NUM_MOE_LAYERS && count < max_prewarm; ++l) {
            for (uint32_t e = 0; e < vault_hdr.max_experts_per_layer && count < max_prewarm; ++e) {
                if (streamer.prefetch(l, e)) count++;
                // Sleep to trickle-load the cache, keeping the MPMC queue empty 
                // and the NVMe SSD available for immediate priority prompt prefetching!
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
        }
    });
    prewarm_thread.detach();

    // ── Phase 4: Scout init ───────────────────────────────────
    smoe::scout::Scout scout(scout_path, metal);  // scout_path may be nullptr (heuristic only)
    const auto& cfg = scout.config();
    const uint32_t d_model = cfg.d_model;
    const uint32_t num_layers = cfg.num_moe_layers + 1;
    const uint32_t ffn_dim = cfg.ffn_dim;
    const uint32_t shared_dim = 2816; // TODO: dynamically load shared expert dim if needed


    // ── Load vocabulary ───────────────────────────────────────
    load_vocab("vault/vocab.bin");

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
            if (g_vocab_loaded && tok < 102400) {
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
    float* full_kv_cache = smoe::allocate_aligned_float(num_layers * 2 * ATTN_CTX * d_model);
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
    
    // Allocate all on 16KB boundaries to guarantee Metal zero-copy wrapping works perfectly
    size_t heavy_hidden_aligned = (d_model * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_hidden, 16384, heavy_hidden_aligned);
    size_t heavy_normed_aligned = (d_model * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_normed, 16384, heavy_normed_aligned);
    ::posix_memalign((void**)&heavy_qbuf,   16384, 16384);
    ::posix_memalign((void**)&heavy_kbuf,   16384, 16384);
    ::posix_memalign((void**)&heavy_vbuf,   16384, 16384);
    size_t heavy_attn_out_aligned = (d_model * sizeof(float) + 16383) & ~16383; ::posix_memalign((void**)&heavy_attn_out, 16384, heavy_attn_out_aligned);
    
    // Allocate heuristic dense/shared FFN intermediate buffers
    float* l0_gate_out = smoe::allocate_aligned_float(ffn_dim);
    float* l0_up_out   = smoe::allocate_aligned_float(ffn_dim);
    float* shared_gate_out = smoe::allocate_aligned_float(shared_dim);
    float* shared_up_out   = smoe::allocate_aligned_float(shared_dim);

    smoe_metal_register_buffer(metal, heavy_hidden, heavy_hidden_aligned);
    smoe_metal_register_buffer(metal, heavy_normed, heavy_normed_aligned);
    smoe_metal_register_buffer(metal, heavy_qbuf,   16384);
    smoe_metal_register_buffer(metal, heavy_kbuf,   16384);
    smoe_metal_register_buffer(metal, heavy_vbuf,   16384);
    smoe_metal_register_buffer(metal, heavy_attn_out, heavy_attn_out_aligned);

    uint32_t heavy_cur_token = (prompt_len > 0) ? prompt_tokens[0] : 0;
    uint32_t scout_cur_token = heavy_cur_token;

    // Speculative lookahead queue
    static constexpr uint32_t GEN_LOOKAHEAD_K = 3;
    static constexpr uint32_t MAX_LOOKAHEAD = 64;
    static smoe::scout::ScoutOutput scout_queue[MAX_LOOKAHEAD];
    uint32_t sq_head = 0;
    uint32_t sq_tail = 0;
    uint32_t sq_size = 0;

    struct ActiveExpertsContext {
        const smoe::scout::ScoutOutput* queue;
        uint32_t head;
        uint32_t size;
    };

    auto is_expert_active = [](uint32_t layer_id, uint32_t expert_id, void* ctx) -> bool {
        auto* c = static_cast<ActiveExpertsContext*>(ctx);
        for (uint32_t i = 0; i < c->size; ++i) {
            uint32_t idx = (c->head + i) % MAX_LOOKAHEAD;
            const auto& scout_out = c->queue[idx];
            if (layer_id == 0) continue;
            if (layer_id - 1 < smoe::NUM_MOE_LAYERS) {
                const auto& pred = scout_out.routing[layer_id - 1];
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

        const float* emb = scout.get_embed() + heavy_cur_token * d_model;
        std::memcpy(heavy_hidden, emb, d_model * sizeof(float));

        if (n == 0 && g_debug) {
            std::fprintf(stderr, "\n[Token %u] emb = %.4f %.4f %.4f %.4f\n", heavy_cur_token, heavy_hidden[0], heavy_hidden[1], heavy_hidden[2], heavy_hidden[3]);
        }

        // Clean up any leaked/discarded slots in the streamer
        {
            ActiveExpertsContext act_ctx { scout_queue, sq_head, sq_size };
            streamer.prune_slots(is_expert_active, &act_ctx);
        }

        // If the lookahead queue is empty (due to prompt, divergence, or heavy catching up),
        // the Scout MUST sync with the Heavy model's current token to provide accurate routing.
        if (sq_size == 0) {
            scout_cur_token = heavy_cur_token;
        }

        // ── Phase A: Scout Speculative Lookahead ──────────────
        uint32_t target_sq_size = is_prompt ? std::min<uint32_t>(prompt_len - n, MAX_LOOKAHEAD) : GEN_LOOKAHEAD_K;

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
            for (uint32_t l = 0; l < smoe::NUM_MOE_LAYERS; ++l) {
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

        std::memcpy(heavy_hidden, scout.get_embed() + static_cast<size_t>(heavy_cur_token) * d_model, d_model * sizeof(float));

        for (uint32_t l = 0; l <= smoe::NUM_MOE_LAYERS; ++l) {
            std::memcpy(heavy_normed, heavy_hidden, d_model * sizeof(float));
            smoe::rms_norm(heavy_normed, scout.get_input_norm(l), d_model);

            smoe::matvec(heavy_qbuf, scout.get_q_proj(l), heavy_normed, d_model, d_model);
            smoe::matvec(heavy_kbuf, scout.get_k_proj(l), heavy_normed, d_model, d_model);
            smoe::matvec(heavy_vbuf, scout.get_v_proj(l), heavy_normed, d_model, d_model);

            // Apply RoPE to Q and K (16 heads, 128 dim) - Half-split pairing
            for (uint32_t h = 0; h < 16; ++h) {
                for (uint32_t d = 0; d < 64; ++d) {
                    float freq = 1.0f / std::pow(10000.0f, static_cast<float>(d * 2) / 128.0f);
                    float angle = static_cast<float>(n) * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);
                    
                    float q0 = heavy_qbuf[h * 128 + d];
                    float q1 = heavy_qbuf[h * 128 + d + 64];
                    heavy_qbuf[h * 128 + d]      = q0 * cos_val - q1 * sin_val;
                    heavy_qbuf[h * 128 + d + 64] = q0 * sin_val + q1 * cos_val;
                    
                    float k0 = heavy_kbuf[h * 128 + d];
                    float k1 = heavy_kbuf[h * 128 + d + 64];
                    heavy_kbuf[h * 128 + d]      = k0 * cos_val - k1 * sin_val;
                    heavy_kbuf[h * 128 + d + 64] = k0 * sin_val + k1 * cos_val;
                }
            }

            float* k_cache = full_kv_cache + (l * 2 * ATTN_CTX + 0 * ATTN_CTX) * d_model;
            float* v_cache = full_kv_cache + (l * 2 * ATTN_CTX + 1 * ATTN_CTX) * d_model;
            
            const uint32_t slot = ctx_pos % ATTN_CTX;
            std::memcpy(k_cache + slot * d_model, heavy_kbuf, d_model * sizeof(float));
            std::memcpy(v_cache + slot * d_model, heavy_vbuf, d_model * sizeof(float));
            scout.write_kv_cache(l, slot, heavy_kbuf, heavy_vbuf);

            // Multi-Head Attention CPU
            const float scale = 1.0f / std::sqrt(128.0f);
            const uint32_t valid = (ctx_fill < ATTN_CTX) ? ctx_fill + 1 : ATTN_CTX;
            std::memset(heavy_attn_out, 0, d_model * sizeof(float));

            for (uint32_t h = 0; h < 16; ++h) {
                for (uint32_t i = 0; i < valid; ++i) {
                    uint32_t ki = (ctx_pos - i + ATTN_CTX) % ATTN_CTX;
                    float dot_qk = 0.0f;
                    const float* krow = k_cache + ki * d_model + h * 128;
                    const float* qhead = heavy_qbuf + h * 128;
                    for (uint32_t d = 0; d < 128; ++d) {
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
                    const float* vrow = v_cache + vi * d_model + h * 128;
                    float* out_head = heavy_attn_out + h * 128;
                    for (uint32_t d = 0; d < 128; ++d) {
                        out_head[d] += alpha * vrow[d];
                    }
                }
            }

            // O Proj and Residual
            smoe::matvec(heavy_normed, scout.get_o_proj(l), heavy_attn_out, d_model, d_model);
            for (uint32_t i = 0; i < d_model; ++i) {
                heavy_hidden[i] += heavy_normed[i];
            }

            // FFN Norm
            std::memcpy(heavy_normed, heavy_hidden, d_model * sizeof(float));
            smoe::rms_norm(heavy_normed, scout.get_post_norm(l), d_model);

            // FFN
            if (l == 0) {
                // Dense MLP for Layer 0
                smoe::matvec(l0_gate_out, scout.get_l0_gate(), heavy_normed, ffn_dim, d_model);
                smoe::matvec(l0_up_out, scout.get_l0_up(), heavy_normed, ffn_dim, d_model);
                for (uint32_t i = 0; i < ffn_dim; ++i) {
                    // Silu
                    float val = l0_gate_out[i];
                    l0_gate_out[i] = (val / (1.0f + std::exp(-val))) * l0_up_out[i];
                }
                smoe::matvec(heavy_normed, scout.get_l0_down(), l0_gate_out, d_model, ffn_dim);
                for (uint32_t d = 0; d < d_model; ++d) heavy_hidden[d] += heavy_normed[d];
                if (n == 0 && g_debug) {
                    std::fprintf(stderr, "\n[L0] hidden = %.4f %.4f %.4f %.4f\n", heavy_hidden[0], heavy_hidden[1], heavy_hidden[2], heavy_hidden[3]);
                }
            } else {
                // ── Phase 1: Dispatch Currently Available Routed Experts to GPU ──
                float* routed_out = (float*)__builtin_alloca(d_model * sizeof(float));
                std::memset(routed_out, 0, d_model * sizeof(float));

                const smoe::scout::ExpertPrediction& pred = scout_out.routing[l - 1];
                bool executed[8] = {false};
                uint32_t num_executed = 0;
                uint64_t spin = 0;
                
                void* wait_handles[8] = {nullptr};
                smoe::io::RingSlot* active_slots[8] = {nullptr};
                static float expert_hidden_scratch[8][4096];
                static float expert_output_vec[8][4096];

                // Initial non-blocking pass
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

                            std::memset(expert_hidden_scratch[e], 0, sizeof(expert_hidden_scratch[e]));
                            std::memset(expert_output_vec[e], 0, sizeof(expert_output_vec[e]));

                            wait_handles[e] = smoe_metal_fused_ffn(
                                metal,
                                packed_gate, scales_gate, packed_up, scales_up, packed_down, scales_down,
                                heavy_normed, expert_hidden_scratch[e], expert_output_vec[e],
                                expert_layout.gate_rows, expert_layout.gate_cols,
                                vault_hdr.group_size, vault_hdr.bits, e
                            );
                        }
                        executed[e] = true;
                        num_executed++;
                    }
                }

                // ── Phase 2: Compute Shared Expert on CPU while GPU/SSD works ──
                float* shared_out = (float*)__builtin_alloca(d_model * sizeof(float));
                
                smoe::matvec(shared_gate_out, scout.get_shared_gate(l), heavy_normed, shared_dim, d_model);
                smoe::matvec(shared_up_out, scout.get_shared_up(l), heavy_normed, shared_dim, d_model);
                for (uint32_t i = 0; i < shared_dim; ++i) {
                    float val = shared_gate_out[i];
                    shared_gate_out[i] = (val / (1.0f + std::exp(-val))) * shared_up_out[i];
                }
                smoe::matvec(shared_out, scout.get_shared_down(l), shared_gate_out, d_model, shared_dim);

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

                                std::memset(expert_hidden_scratch[e], 0, sizeof(expert_hidden_scratch[e]));
                                std::memset(expert_output_vec[e], 0, sizeof(expert_output_vec[e]));

                                wait_handles[e] = smoe_metal_fused_ffn(
                                    metal,
                                    packed_gate, scales_gate, packed_up, scales_up, packed_down, scales_down,
                                    heavy_normed, expert_hidden_scratch[e], expert_output_vec[e],
                                    expert_layout.gate_rows, expert_layout.gate_cols,
                                    vault_hdr.group_size, vault_hdr.bits, e
                                );
                            }
                            executed[e] = true;
                            num_executed++;
                            made_progress = true;
                        } else {
                            if (spin % 100000 == 0) streamer.prefetch(pred.layer_id, pred.expert_ids[e]);
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
            }
        }
        
        ctx_pos = (ctx_pos + 1) % ATTN_CTX;
        if (ctx_fill < ATTN_CTX) ++ctx_fill;

        // ── Step 5: Final Model Norm and LM Head ──────────────
        smoe::rms_norm(heavy_hidden, scout.get_model_norm(), d_model);
        
        // Execute LM Head on GPU
        const float* weights[1] = { scout.get_lm_head() };
        const float* inputs[1]  = { heavy_hidden };
        float* outputs[1]       = { scout.get_lm_head_scores() };
        uint32_t rows[1]        = { 102400 };
        uint32_t cols[1]        = { d_model };
        smoe_metal_scout_matvec_batch(metal, weights, inputs, outputs, rows, cols, 1);

        // Process lm_head result
        float* scores = scout.get_lm_head_scores();
        uint32_t best_tok   = 0;
        
        // Repetition penalty
        if (rep_penalty != 1.0f) {
            bool penalized[102400] = {false};
            uint32_t penalty_last_n = 256; // 256 tokens is enough to break loops without forcing language switches
            uint32_t start_idx = (history_len > penalty_last_n) ? (history_len - penalty_last_n) : 0;
            
            for (uint32_t i = start_idx; i < history_len; ++i) {
                uint32_t tok = token_history[i];
                // Exempt common punctuation and special tokens from penalty
                // 13: '.', 11: ',', 185: '\n', 30: '?', 0: '!', 100000: BOS, 100001: EOS
                if (tok == 13 || tok == 11 || tok == 185 || tok == 30 || tok == 0 || tok == 100000 || tok == 100001) {
                    continue;
                }
                if (!penalized[tok]) {
                    if (scores[tok] <= 0) {
                        scores[tok] *= rep_penalty;
                    } else {
                        scores[tok] /= rep_penalty;
                    }
                    penalized[tok] = true;
                }
            }
        }

        if (temperature < 1e-4f) {
            // Greedy
            float best_score = -1e38f;
            for (uint32_t v = 0; v < 102400; ++v) {
                if (scores[v] > best_score) {
                    best_score = scores[v];
                    best_tok   = v;
                }
            }
            if (g_debug && ((prompt_len == 0) || (n >= prompt_len - 1))) {
                float h_sum2 = 0.0f;
                for(uint32_t i=0; i<d_model; i++) h_sum2 += heavy_hidden[i]*heavy_hidden[i];
                std::fprintf(stderr, "\n[n=%u] hidden_L2 = %f, Top score = %f for tok = %u\n", n, std::sqrt(h_sum2), best_score, best_tok);
            }
        } else {
            // Temperature + Top-p
            float max_score = scores[0];
            for (uint32_t v = 1; v < 102400; ++v) {
                if (scores[v] > max_score) max_score = scores[v];
            }
            
            float sum_exp = 0.0f;
            struct ProbTok { float p; uint32_t v; };
            static ProbTok probs[102400];
            
            for (uint32_t v = 0; v < 102400; ++v) {
                probs[v].v = v;
                probs[v].p = std::exp((scores[v] - max_score) / temperature);
                sum_exp += probs[v].p;
            }
            
            float inv_sum = 1.0f / sum_exp;
            for (uint32_t v = 0; v < 102400; ++v) {
                probs[v].p *= inv_sum;
            }
            
            std::sort(probs, probs + 102400, [](const ProbTok& a, const ProbTok& b) {
                return a.p > b.p;
            });
            
            float cumsum = 0.0f;
            uint32_t top_k_len = 0;
            for (uint32_t v = 0; v < 102400; ++v) {
                cumsum += probs[v].p;
                top_k_len++;
                if (cumsum >= top_p || top_k_len >= top_k) break;
            }
            
            std::uniform_real_distribution<float> dist(0.0f, cumsum);
            float r = dist(rng);
            float running = 0.0f;
            best_tok = probs[top_k_len - 1].v;
            for (uint32_t v = 0; v < top_k_len; ++v) {
                running += probs[v].p;
                if (r <= running) {
                    best_tok = probs[v].v;
                    break;
                }
            }
        }
        
        heavy_cur_token = best_tok;
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
            
            // Immediately prune discarded experts
            ActiveExpertsContext act_ctx { scout_queue, sq_head, sq_size };
            streamer.prune_slots(is_expert_active, &act_ctx);
        } else {
            // Scout guessed correctly (or we are in prompt phase)
            // Consume the head of the queue.
            sq_head = (sq_head + 1) % MAX_LOOKAHEAD;
            sq_size--;
        }

        // ── Step 6: Emit token ────────────────────────────────
        if (is_generating) {
            if (heavy_cur_token == 100001) {
                if (g_debug) std::fprintf(stderr, "\n[EOS reached]\n");
                break;
            }
            if (g_raw_ids) {
                std::printf("%u ", heavy_cur_token);
            } else if (g_vocab_loaded && heavy_cur_token < 102400) {
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
