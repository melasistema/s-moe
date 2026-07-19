#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

namespace smoe {
namespace cli {

inline constexpr uint32_t DEFAULT_RING_SIZE    = 0;
inline constexpr uint32_t DEFAULT_WORKERS      = 16;
inline constexpr uint64_t DEFAULT_SLOT_MB      = 0;
inline constexpr uint32_t DEFAULT_MAX_TOKENS   = 512;

struct EngineConfig {
    const char* vault_path  = nullptr;
    const char* scout_path  = nullptr;
    const char* tokens_in   = nullptr;
    uint32_t    max_tokens  = DEFAULT_MAX_TOKENS;
    uint32_t    ring_size   = DEFAULT_RING_SIZE;
    uint32_t    num_workers = DEFAULT_WORKERS;
    uint64_t    slot_mb     = DEFAULT_SLOT_MB;
    float       temperature = 0.6f;
    float       top_p       = 0.95f;
    uint32_t    top_k       = 50;
    float       rep_penalty = 1.0f;
    std::vector<uint32_t> eos_ids = {151645, 151643};
    uint32_t    spec_width  = 0;   // extra gate ranks (9..8+N) prefetched
                                   // speculatively for the next token; 0 = off.
                                   // Measured NET-NEGATIVE at Q4 (spec 8:
                                   // +242ms dense from memory-bandwidth
                                   // contention, +144ms io-spin from worker
                                   // occupancy) despite lifting ring coverage
                                   // 46→64%. Re-test with Q2 vaults: half the
                                   // bytes, double the idle bandwidth.
    bool        debug       = false;
    bool        raw_ids     = false;
    bool        serve       = false;
    bool        instrument  = false;
    bool        valid       = true;
};

static inline void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "\n"
        "  S-MoE Engine — Streaming Mixture-of-Experts Inference\n"
        "  ───────────────────────────────────────────────────────\n"
        "  Usage:\n"
        "    %s --vault <path.smoe> --scout <path> (--tokens-in <ids> | --serve) [options]\n"
        "\n"
        "  Required:\n"
        "    --vault     <path>  Path to the .smoe vault file\n"
        "    --scout     <path>  Path to the dense-backbone .safetensors weights\n"
        "  and one input mode:\n"
        "    --tokens-in <ids>   Comma-separated prompt token IDs (one-shot run)\n"
        "    --serve             Persistent server mode: read GEN/RESET requests\n"
        "                        on stdin, reuse KV cache across requests\n"
        "                        (chat.py / serve.py drive this mode)\n"
        "\n"
        "  Optional:\n"
        "    --tokens  <N>       Max tokens to generate (default: %u)\n"
        "    --ring    <N>       Ring buffer slot count  (default: %u = auto)\n"
        "    --workers <N>       I/O worker thread count (default: %u)\n"
        "    --slot-mb <N>       Bytes per ring slot, MB (default: %llu = auto)\n"
        "    --raw-ids           Print raw token IDs as integers instead of text\n"
        "    --instrument        Print a per-request decode timing breakdown\n"
        "                        (dense/io/gpu buckets) on stderr\n"
        "\n"
        "  Example:\n"
        "    %s --vault vault/qwen3-30b-instruct.smoe \\\n"
        "       --scout vault/qwen3-30b-instruct.scout.safetensors --serve\n"
        "\n",
        argv0,
        DEFAULT_MAX_TOKENS,
        DEFAULT_RING_SIZE,
        DEFAULT_WORKERS,
        DEFAULT_SLOT_MB,
        argv0);
}

static inline EngineConfig parse_args(int argc, char* argv[]) {
    EngineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* flag) {
            return std::strcmp(argv[i], flag) == 0 && i + 1 < argc;
        };
        if      (arg("--vault"))     { cfg.vault_path  = argv[++i]; }
        else if (arg("--scout"))     { cfg.scout_path  = argv[++i]; }
        else if (arg("--tokens-in")) { cfg.tokens_in   = argv[++i]; }
        else if (arg("--tokens"))    { cfg.max_tokens  = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--ring"))      { cfg.ring_size   = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--workers"))   { cfg.num_workers = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--slot-mb"))   { cfg.slot_mb     = static_cast<uint64_t>(std::atoll(argv[++i])); }
        else if (arg("--temperature")){ cfg.temperature = static_cast<float>(std::atof(argv[++i])); }
        else if (arg("--top-p"))      { cfg.top_p = static_cast<float>(std::atof(argv[++i])); }
        else if (arg("--top-k"))      { cfg.top_k = static_cast<uint32_t>(std::atoi(argv[++i])); }
        else if (arg("--rep-penalty")){ cfg.rep_penalty = static_cast<float>(std::atof(argv[++i])); }
        else if (arg("--spec"))       { cfg.spec_width = static_cast<uint32_t>(std::atoi(argv[++i]));
                                        if (cfg.spec_width > 24) cfg.spec_width = 24; }
        else if (arg("--eos-ids")) {
            cfg.eos_ids.clear();
            std::string ids_str = argv[++i];
            size_t pos = 0;
            while ((pos = ids_str.find(',')) != std::string::npos) {
                if (pos > 0) cfg.eos_ids.push_back(std::stoul(ids_str.substr(0, pos)));
                ids_str.erase(0, pos + 1);
            }
            if (!ids_str.empty()) cfg.eos_ids.push_back(std::stoul(ids_str));
        }
        else if (std::strcmp(argv[i], "--debug") == 0) { cfg.debug = true; }
        else if (std::strcmp(argv[i], "--raw-ids") == 0) { cfg.raw_ids = true; }
        else if (std::strcmp(argv[i], "--serve") == 0) { cfg.serve = true; }
        else if (std::strcmp(argv[i], "--instrument") == 0) { cfg.instrument = true; }
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h")     == 0) {
            print_usage(argv[0]);
            cfg.valid = false;
            return cfg;
        }
    }

    if (!cfg.vault_path || (!cfg.tokens_in && !cfg.serve)) {
        std::fprintf(stderr, "\n  ✗  --vault and either --tokens-in or --serve are required.\n");
        print_usage(argv[0]);
        cfg.valid = false;
        return cfg;
    }

    const char* sm_dbg = std::getenv("SMOE_DEBUG");
    if (sm_dbg) {
        cfg.debug = cfg.debug || (std::strcmp(sm_dbg, "1") == 0 || std::strcmp(sm_dbg, "true") == 0);
    }

    return cfg;
}

} // namespace cli
} // namespace smoe
