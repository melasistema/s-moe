// ═══════════════════════════════════════════════════════════════
// main.cpp — S-MoE Engine · Core Loop & Live Telemetry
// ═══════════════════════════════════════════════════════════════
// Responsibilities:
//   ① CLI argument parsing and validation
//   ② Pre-flight vault inspection (SmoeHeader + v2 arch block)
//   ③ Request loop (one-shot --tokens-in, or --serve GEN/RESET):
//        layer-major prefill → per-token decode with exact routing,
//        demand/speculative expert prefetch, Metal FFN → emit token
//   ④ Live telemetry bar: t/s | RAM | NVMe GB/s | miss %
//   ⑤ Clean shutdown: drain ring, join workers, free Metal
//
// Design invariants (CLAUDE.md):
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
#include "engine.hpp"
#include "decode.hpp"
#include "serve.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// ── Compile-time defaults ─────────────────────────────────────
inline constexpr uint32_t TELEMETRY_EVERY      = 10;   // tokens between telemetry updates

using smoe::engine::wall_ms;
using smoe::engine::resident_bytes;

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

// Reads the 64-byte header, and — on a v2 vault — the SmoeArchBlock it points
// to. `arch_valid` reports whether `arch` holds a validated block; a v1 vault
// returns true with arch_valid=false (legacy config-inference path).
static bool read_vault_header(const char* vault_path, smoe::SmoeHeader& hdr,
                              smoe::SmoeArchBlock& arch, bool& arch_valid) {
    arch_valid = false;
    int fd = ::open(vault_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "\n  ✗  Cannot open vault: %s\n", vault_path);
        return false;
    }

    ssize_t n = ::read(fd, &hdr, sizeof(smoe::SmoeHeader));

    if (n != ssize_t(sizeof(smoe::SmoeHeader))) {
        ::close(fd);
        std::fprintf(stderr, "\n  ✗  Vault too small (not a valid .smoe file).\n");
        return false;
    }
    if (!smoe::magic_valid(hdr.magic)) {
        ::close(fd);
        std::fprintf(stderr, "\n  ✗  Invalid magic bytes — not a .smoe vault.\n");
        return false;
    }
    if (hdr.version < smoe::SMOE_VERSION_MIN || hdr.version > smoe::SMOE_VERSION) {
        ::close(fd);
        std::fprintf(stderr,
            "\n  ✗  Vault version mismatch: got %u, supported %u–%u.\n",
            hdr.version, smoe::SMOE_VERSION_MIN, smoe::SMOE_VERSION);
        return false;
    }

    if (hdr.version >= 2) {
        // v2 requires a valid arch block; a missing/corrupt one is a hard
        // error rather than a silent fall-back to guessed constants.
        if (hdr.reserved_ext == 0 ||
            hdr.reserved_ext + sizeof(smoe::SmoeArchBlock) > hdr.data_offset) {
            ::close(fd);
            std::fprintf(stderr, "\n  ✗  v2 vault has no valid arch-block offset.\n");
            return false;
        }
        ssize_t an = ::pread(fd, &arch, sizeof(arch), hdr.reserved_ext);
        ::close(fd);
        if (an != ssize_t(sizeof(arch)) || !smoe::arch_magic_valid(arch.magic) ||
            arch.block_size != sizeof(smoe::SmoeArchBlock) ||
            arch.block_version != smoe::SMOE_ARCH_VERSION) {
            std::fprintf(stderr, "\n  ✗  Corrupt arch block in v2 vault.\n");
            return false;
        }
        if (arch.activation != smoe::SMOE_ACT_SILU) {
            std::fprintf(stderr,
                "\n  ✗  Vault requires activation id %u; engine implements SiLU only.\n",
                arch.activation);
            return false;
        }
        if (arch.moe_top_k == 0 || arch.moe_top_k > smoe::scout::MAX_ACTIVE ||
            arch.head_dim == 0 || arch.head_dim > 256 ||
            arch.num_heads == 0 || arch.num_kv_heads == 0 ||
            arch.num_heads % arch.num_kv_heads != 0) {
            std::fprintf(stderr, "\n  ✗  Arch block values out of engine limits "
                "(top_k=%u heads=%u/%u head_dim=%u).\n",
                arch.moe_top_k, arch.num_heads, arch.num_kv_heads, arch.head_dim);
            return false;
        }
        arch_valid = true;
        return true;
    }

    ::close(fd);
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

static bool g_debug = false;
static bool g_raw_ids = false;
static bool g_instr = false;

// ── Expert popularity histogram (prewarm ordering) ────────────
// Claim counts per (absolute layer, expert), persisted via
// engine::load/save_expert_freq across runs.
static uint32_t g_expert_hits[128 * smoe::prefill::HITS_STRIDE];

// ── Main entry point ──────────────────────────────────────────

int main(int argc, char* argv[]) {

    // ── Argument parsing ──────────────────────────────────────
    smoe::cli::EngineConfig cli_cfg = smoe::cli::parse_args(argc, argv);
    if (!cli_cfg.valid) return 1;

    const char* vault_path  = cli_cfg.vault_path;
    const char* scout_path  = cli_cfg.scout_path;
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
    smoe::SmoeArchBlock vault_arch {};
    bool vault_arch_valid = false;
    if (!read_vault_header(vault_path, vault_hdr, vault_arch, vault_arch_valid)) return 1;

    if (g_debug) {
        std::fprintf(stderr,
            "\n"
            "  S-MoE Engine\n"
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

    // ── Auto-compute slot size and ring size (see engine.cpp) ──
    smoe::engine::autotune(vault_path, scout_path, vault_hdr, slot_mb, ring_size);

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
    smoe::scout::Scout scout(scout_path, metal, &vault_hdr,
                             vault_arch_valid ? &vault_arch : nullptr);  // scout_path may be nullptr (heuristic only)
    const auto& cfg = scout.config();

    // ── Popularity-Ordered Cache Pre-Warming ──────────────────
    // Streams the historically hottest experts into the ring while the
    // engine is idle: at startup and, in serve mode, between requests
    // (rearm() re-arms a pass, set_busy() pauses it so it never
    // competes with demand reads). See engine.cpp.
    smoe::engine::load_expert_freq(g_expert_hits);
    const uint32_t prewarm_start_layer = cfg.has_dense_layer_0 ? 1 : 0;
    smoe::engine::Prewarm prewarm;
    prewarm.start(&streamer, g_expert_hits, prewarm_start_layer,
                  cfg.num_moe_layers, vault_hdr.max_experts_per_layer);

    const uint32_t d_model = cfg.d_model;
    const uint32_t moe_start_layer = cfg.has_dense_layer_0 ? 1 : 0;

    // ── Load vocabulary ───────────────────────────────────────
    const bool vocab_loaded =
        smoe::engine::load_vocab(smoe::engine::VOCAB_BIN_PATH, cfg.vocab_size, g_debug);

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
    // Sized to the serve-protocol ceiling (k=<N> per-request override), not
    // the launch --top-k, so overrides never reallocate mid-loop (Golden
    // Rule: no dynamic memory in the token loop). k requests above the cap
    // are clamped at parse time.
    constexpr uint32_t TOP_K_PROTOCOL_CAP = 1024;
    smoe::SamplerScratch sampler_scratch(std::max(top_k, TOP_K_PROTOCOL_CAP), cfg.vocab_size);
    TelemetryState ts;

    // Instrumentation state lives in decode::Instr; decode charges the
    // in-layer buckets, main charges lm-head and the per-token totals.
    smoe::decode::Instr instr {};
    bool instr_gen = false;

    // ── Token Generation Loop ─────────────────────────────────
    //
    // Per token step: exact routing at every MoE layer (real gate
    // matvec on the heavy hidden state), demand prefetch of the
    // top-8, speculative prefetch of the deeper gate ranks.

    constexpr uint32_t ATTN_CTX = smoe::decode::ATTN_CTX;
    uint32_t ctx_pos = 0;
    uint32_t ctx_fill = 0;

    // ── Pre-allocated, Metal-registered activation buffers ────
    // Every plane the forward pass touches, allocated once (engine.cpp).
    smoe::engine::Buffers buf;
    buf.init(metal, cfg, vault_hdr, ATTN_CTX);

    // Exact routing happens inside decode::run_token_step: the real
    // router gate is evaluated on the heavy hidden state at every MoE
    // layer, in prefill AND decode. Ranks beyond top-k (up to --spec
    // extra) double as the next-token speculative prefetch.
    const uint32_t spec_width = cli_cfg.spec_width;
    // NOTE: replaying the previous token's routing as PREFILL prefetch
    // hints was tried and measured SLOWER (114s vs 84s TTFT on the
    // 45-token bench) — hints added wrong reads ahead of demand fetches.
    // Prefill I/O overlap comes from layer-major batched prefill instead.
    // Decode is different: measured adjacent-token expert overlap there
    // is ~53%, which ring retention exploits for free (it skips reads
    // rather than adding them).

    // NOTE: the decode-time scout speculation (Phase A lookahead queue +
    // divergence rollback) was removed after measurement: the scout's
    // full backbone pass cost ~265 ms/token, and its routing predictions
    // matched the true exact-gate routing only 51.5% of the time —
    // barely above the 46.4% the retained ring already provides for
    // free. The decode prefetch oracle is now the previous token's own
    // top-16 gate ranking (measured 63.7% coverage of the next token's
    // true top-8), fired through the streamer's low-priority queue.
    uint32_t next_heavy_token = 0;

    smoe::decode::Params dparams {
        .cfg = &cfg,
        .scout = &scout,
        .streamer = &streamer,
        .metal = metal,
        .hdr = &vault_hdr,
        .layout = &expert_layout,
        .b = &buf,
        .moe_start_layer = moe_start_layer,
        .spec_width = spec_width,
        .expert_hits = g_expert_hits,
        .debug = g_debug,
    };

    const bool serve = cli_cfg.serve;
    bool one_shot_done = false;
    std::string req_line; // reused across requests; allocation stays out of the token loop

    // ═══ REQUEST LOOP ═══════════════════════════════════════════
    // One-shot mode runs exactly one request from the CLI arguments.
    // Server mode reads one request per stdin line:
    //   GEN <max_tokens> [ovr ...] <id0>,<id1>,...   full conversation stream
    //   RESET                                        drop all cached state
    // [ovr ...] are optional per-request sampling overrides, space-separated
    // key=value fields: t=<temperature> p=<top_p> k=<top_k> r=<rep_penalty>.
    // The first field without '=' starts the token csv, so pre-override
    // clients (chat.py) are parsed unchanged. Unknown keys are ignored.
    // The engine replies with raw token IDs on stdout, terminated by <<DONE>>.
    // The KV cache, ring cache, and scout state persist across requests;
    // only the suffix beyond the longest common prefix with the stored
    // stream is prefetched and prefilled.
    while (true) {
        uint32_t req_max_tokens = max_tokens;
        // Per-request sampling: launch values are the defaults, serve-mode
        // overrides apply to this request only.
        float    req_temperature = temperature;
        float    req_top_p       = top_p;
        uint32_t req_top_k       = top_k;
        float    req_rep_penalty = rep_penalty;
        prompt_len = 0;

        if (!serve) {
            if (one_shot_done) break;
            one_shot_done = true;
            // One-shot mode always has --tokens-in (enforced at CLI parse).
            {
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
                std::printf("<<DONE>>\n");
                std::fflush(stdout);
                continue;
            }
            if (req_line.rfind("GEN ", 0) != 0) {
                std::printf("<<ERR bad_request>>\n");
                std::fflush(stdout);
                continue;
            }

            // Parse: GEN <max_tokens> [ovr ...] <csv-ids>  (serve.hpp)
            static uint32_t full_toks[STREAM_CAP];
            smoe::serve::GenRequest req {};
            req.max_tokens  = req_max_tokens;
            req.temperature = req_temperature;
            req.top_p       = req_top_p;
            req.top_k       = req_top_k;
            req.rep_penalty = req_rep_penalty;
            if (!smoe::serve::parse_gen(req_line, req, full_toks, STREAM_CAP,
                    static_cast<uint32_t>(sampler_scratch.topk.size()))) {
                std::printf("<<ERR %s>>\n",
                            req.count >= STREAM_CAP ? "ctx_overflow" : "bad_request");
                std::fflush(stdout);
                continue;
            }
            req_max_tokens  = req.max_tokens;
            req_temperature = req.temperature;
            req_top_p       = req.top_p;
            req_top_k       = req.top_k;
            req_rep_penalty = req.rep_penalty;
            const uint32_t full_len = req.count;

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
                std::fprintf(stderr, "[serve] stream=%u lcp=%u suffix=%u gen=%u t=%.2f p=%.2f k=%u r=%.2f\n",
                             full_len, lcp, suffix, req_max_tokens,
                             req_temperature, req_top_p, req_top_k, req_rep_penalty);
            }
        }

        // Absolute position of prompt_tokens[0] in the token stream —
        // RoPE and the scout sync must use stream positions, not the
        // per-request loop index.
        const uint64_t stream_base = stream_len - prompt_len;

        // Pause the prewarm trickle for the duration of the request so
        // demand reads own the NVMe queue.
        prewarm.set_busy(true);

        // ── Per-request state ─────────────────────────────────
        ts.start_ms   = wall_ms();
        ts.prev_ms    = ts.start_ms;
        ts.prev_bytes = 0;

        next_heavy_token = 0;

        // Finish reason for serve clients: EOS sampled ("stop") vs the
        // req_max_tokens cap exhausted ("length").
        bool hit_eos = false;

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
                .full_kv_cache = buf.kv_cache,
                .attn_ctx = ATTN_CTX,
                .moe_start_layer = moe_start_layer,
                .rope_inv_freq = buf.rope_inv_freq,
                .b = buf.pf,
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
            it_tok0    = wall_ms();
            instr.mark = it_tok0;
            it_bytes0  = streamer.bytes_read();
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

        // Token emission happens once in Step 6; anything here is debug telemetry
        // only and must stay off stdout (chat.py parses stdout as the token stream).
        if (n >= prompt_len && g_debug) {
            std::fprintf(stderr, "[RAW TOK: %u] ", heavy_cur_token);
        }
        if (n == 0) {
            std::fprintf(stderr, "\n");
        }

        // One token through all layers (decode.cpp); the final hidden
        // state lands in buf.hidden for the LM head below.
        smoe::decode::run_token_step(dparams, g_instr ? &instr : nullptr,
                                     heavy_cur_token, stream_base + n,
                                     is_prompt, /*first_step=*/n == 0,
                                     ctx_pos, ctx_fill);

        // ── Step 5: Final Model Norm and LM Head ──────────────
        // Prompt positions before the last already know their next token,
        // so the vocab-sized LM head matvec (~622M MACs on CPU) and the
        // sampling pass would be discarded work — skip them entirely.
        // buf.hidden is rebuilt from the embedding at the top of every
        // step, so leaving it un-normed here is safe.
        bool is_generating = (prompt_len == 0) || (n >= prompt_len - 1);
        if (is_generating) {
            if (g_debug) {
                float h_sum2 = 0.0f;
                for(uint32_t i=0; i<d_model; i++) h_sum2 += buf.hidden[i]*buf.hidden[i];
                std::fprintf(stderr, "\n[n=%u] raw_hidden_L2 = %f\n", n, std::sqrt(h_sum2));
            }
            smoe::rms_norm_bf16(buf.hidden, scout.get_model_norm(), d_model);

            // LM Head: vocab × d_model matvec — GPU when available (the
            // scout already registered w_lm_head and lm_head_scores with
            // Metal; buf.hidden is registered at init).
            float* scores = scout.get_lm_head_scores();
            if (metal) {
                smoe_metal_scout_matvec_bf16(metal, scout.get_lm_head(), buf.hidden, scores, cfg.vocab_size, d_model);
            } else {
                smoe::matvec_bf16(scores, scout.get_lm_head(), buf.hidden, cfg.vocab_size, d_model);
            }

            // Sample Token
            smoe::SamplerConfig sampler_cfg {
                .vocab_size = cfg.vocab_size,
                .temperature = req_temperature,
                .top_p = req_top_p,
                .top_k = req_top_k,
                .rep_penalty = req_rep_penalty
            };
            uint32_t best_tok = smoe::sample_token(scores, sampler_cfg, stream_tokens, stream_len, rng, sampler_scratch);

            heavy_cur_token = best_tok;
            if (g_debug) {
                float h_sum2 = 0.0f;
                for(uint32_t i=0; i<d_model; i++) h_sum2 += buf.hidden[i]*buf.hidden[i];
                std::fprintf(stderr, "\n[n=%u] hidden_L2 = %f, best_tok = %u\n", n, std::sqrt(h_sum2), best_tok);
                std::fflush(stderr);
            }
            next_heavy_token = best_tok; // Save for the next contiguous iteration
            // NOTE: best_tok is NOT appended to stream_tokens here — it
            // enters the stream at the top of the step that processes it.
        }
        if (instr_gen) instr.bucket(instr.lm_ms);

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
                hit_eos = true;
                break;
            }
            if (g_raw_ids) {
                std::printf("%u ", heavy_cur_token);
            } else if (vocab_loaded && heavy_cur_token < cfg.vocab_size) {
                std::string s = smoe::engine::vocab()[heavy_cur_token];
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
        if (g_instr) instr.print_and_reset();

        // Resume prewarm with the freshly updated popularity histogram:
        // it re-warms the ring while the user types the next message.
        smoe::engine::save_expert_freq(g_expert_hits);
        prewarm.set_busy(false);
        prewarm.rearm();

        if (serve) {
            // Terminate the token stream for this request. Clients read
            // token IDs until the <<DONE>> sentinel. The fin=<reason>
            // key=value token rides ahead of it (mirroring the GEN
            // override syntax): "stop" = EOS sampled, "length" = the
            // max_tokens cap ran out. Backward compatible — clients that
            // predate it (chat.py) drop non-numeric tokens unread.
            std::printf("\nfin=%s <<DONE>>\n", hit_eos ? "stop" : "length");
            std::fflush(stdout);
        }
    } // ═══ end REQUEST LOOP ═══


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
    prewarm.stop();
    smoe::engine::save_expert_freq(g_expert_hits);
    streamer.shutdown();
    smoe_metal_destroy(metal);

    return 0;
}
