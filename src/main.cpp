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

// ── Compile-time defaults ─────────────────────────────────────
inline constexpr uint32_t DEFAULT_RING_SIZE    = 48;
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
    uint32_t    max_tokens  = DEFAULT_MAX_TOKENS;
    uint32_t    ring_size   = DEFAULT_RING_SIZE;
    uint32_t    num_workers = DEFAULT_WORKERS;
    uint64_t    slot_mb     = DEFAULT_SLOT_MB;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if      (arg("--vault"))   { vault_path  = argv[++i]; }
        else if (arg("--scout"))   { scout_path  = argv[++i]; }
        else if (arg("--prompt"))  { prompt_text = argv[++i]; }
        else if (arg("--tokens"))  { max_tokens  = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--ring"))    { ring_size   = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--workers")) { num_workers = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--slot-mb")) { slot_mb     = static_cast<uint64_t>(std::atoll(argv[++i])); }
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h")     == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!vault_path || !prompt_text) {
        std::fprintf(stderr,
            "\n  ✗  --vault and --prompt are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    // ── Vault header read ─────────────────────────────────────
    smoe::SmoeHeader vault_hdr {};
    if (!read_vault_header(vault_path, vault_hdr)) return 1;

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

    // ── Phase 3: Metal bridge init ────────────────────────────
    uint64_t slot_bytes = slot_mb * 1024ULL * 1024ULL;
    SmoeMetalCtx* metal = smoe_metal_init(slot_bytes);
    if (!metal) {
        std::fprintf(stderr, "  ✗  Metal initialisation failed.\n");
        return 1;
    }

    // ── Phase 2: Streamer init ────────────────────────────────
    smoe::io::Streamer streamer(vault_path, ring_size, num_workers, slot_bytes);

    // ── Phase 4: Scout init ───────────────────────────────────
    smoe::scout::Scout scout(scout_path);  // scout_path may be nullptr (heuristic only)

    // ── Tokenise prompt ───────────────────────────────────────
    // Pre-allocate on the stack — no heap in the loop.
    static uint32_t prompt_tokens[2048];
    uint32_t        prompt_len = 0;
    tokenise_prompt(prompt_text, prompt_tokens, prompt_len,
                    std::min(max_tokens, uint32_t(2048)));

    // ── Pre-generate telemetry state ──────────────────────────
    TelemetryState ts;
    ts.start_ms   = wall_ms();
    ts.prev_ms    = ts.start_ms;
    ts.prev_bytes = 0;

    // ── Print prompt passthrough ──────────────────────────────
    std::fprintf(stdout, "\n");
    for (uint32_t i = 0; i < prompt_len; ++i) {
        // Heuristic "detokeniser": treat token ID as ASCII code point.
        // Week 5+: replace with real BPE decoder.
        unsigned char c = static_cast<unsigned char>(prompt_tokens[i] & 0xFF);
        std::fputc(c, stdout);
    }

    // ── Token Generation Loop ─────────────────────────────────
    //
    // Invariant per token step N:
    //
    //   1. Scout.forward(token) → next_token + K-step lookahead
    //   2. Fire prefetch() for all predicted experts in lookahead[0..K-1]
    //   3. Claim one READY ring slot for the current step's expert
    //   4. Dispatch Metal FFN kernel on claimed slot
    //   5. Release slot back to ring
    //   6. Emit next_token to stdout
    //   7. Every TELEMETRY_EVERY tokens: update telemetry bar

    uint32_t cur_token = (prompt_len > 0) ? prompt_tokens[prompt_len - 1] : 0;

    for (uint32_t n = 0; n < max_tokens; ++n) {

        // ── Step 1: Scout forward ─────────────────────────────
        smoe::scout::ScoutOutput scout_out = scout.forward(cur_token);

        // ── Step 2: Prefetch predicted experts ────────────────
        for (uint32_t k = 0; k < smoe::scout::LOOKAHEAD_K; ++k) {
            const smoe::scout::ExpertPrediction& pred = scout_out.lookahead[k];
            for (uint32_t e = 0; e < pred.count; ++e) {
                // Non-blocking — returns false under back-pressure; that's fine.
                (void)streamer.prefetch(pred.layer_id, pred.expert_ids[e]);
            }
        }

        // ── Step 3: Claim a READY slot ────────────────────────
        // Spin briefly for a READY slot.  If none is available,
        // the Scout's lookahead was inaccurate (miss) — we yield
        // for one scheduling quantum and try again.
        smoe::io::RingSlot* slot = nullptr;
        for (int spin = 0; spin < 1000 && !slot; ++spin) {
            slot = streamer.claim_ready();
            if (!slot) std::this_thread::yield();
        }

        if (slot) {
            // ── Step 4: Metal FFN dispatch ────────────────────
            // In Week 4 with the heuristic Scout, the slot data
            // is whatever the Streamer loaded.  We dispatch the
            // Metal kernel to validate the dispatch pipeline.
            //
            // The slot->data pointer is directly into UMA, so
            // we can use it as input to the GPU without copying.
            //
            // For the heuristic validation build, we pass
            // slot->data as a unified input/output pointer pair
            // and dummy dims.  Week 5+ uses real TensorDescriptors.
            if (slot->data_size >= 64) {  // sanity: at least one group
                // Minimal dispatch: use slot memory as gate/up/down
                // all pointing to the same blob, with toy dimensions.
                // This exercises the full Metal dispatch pipeline.
                const uint32_t dummy_rows = 16;
                const uint32_t dummy_cols = 16;

                // Scratch output vector (stack — not in alloc-banned zone
                // as it's tiny and compiler-placed)
                static float dummy_hidden[64];
                static float dummy_output[64];
                static float dummy_input [64];
                std::memset(dummy_input,  0, sizeof(dummy_input));
                std::memset(dummy_hidden, 0, sizeof(dummy_hidden));

                smoe_metal_fused_ffn(
                    metal,
                    slot->data,                               // gate packed
                    reinterpret_cast<const uint16_t*>(       // gate scales
                        slot->data + dummy_rows * dummy_cols / 4),
                    slot->data,                               // up packed (same for validation)
                    reinterpret_cast<const uint16_t*>(
                        slot->data + dummy_rows * dummy_cols / 4),
                    slot->data,                               // down packed
                    reinterpret_cast<const uint16_t*>(
                        slot->data + dummy_rows * dummy_cols / 4),
                    dummy_input,
                    dummy_hidden,
                    dummy_output,
                    dummy_rows, dummy_cols,
                    smoe::Q2_GROUP_SIZE);
            }

            // ── Step 5: Release slot ──────────────────────────
            streamer.release(slot);
            smoe_metal_swap_buffers(metal);
        }
        // If no slot was ready, we count it as a miss — the Streamer
        // telemetry counters already track this internally.

        // ── Step 6: Emit token ────────────────────────────────
        cur_token = scout_out.next_token_id;
        unsigned char c = static_cast<unsigned char>(cur_token & 0xFF);
        std::fputc(c, stdout);
        std::fflush(stdout);

        // ── Step 7: Telemetry ─────────────────────────────────
        if ((n + 1) % TELEMETRY_EVERY == 0) {
            print_telemetry(n + 1, streamer, ts);
        }
    }

    // Final telemetry flush
    print_telemetry(max_tokens, streamer, ts);
    std::fprintf(stderr, "\n\n");

    // ── Shutdown ──────────────────────────────────────────────
    streamer.shutdown();
    smoe_metal_destroy(metal);

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

    return 0;
}
