// ═══════════════════════════════════════════════════════════════
// serve.hpp — S-MoE Engine · Serve-Mode Line Protocol
// ═══════════════════════════════════════════════════════════════
// Parsing for the --serve stdin protocol (one request per line):
//
//   GEN <max_tokens> [ovr ...] <id0>,<id1>,...   full conversation stream
//   RESET                                        drop all cached state
//
// [ovr ...] are optional per-request sampling overrides, space-
// separated key=value fields: t=<temperature> p=<top_p> k=<top_k>
// r=<rep_penalty>. The first field without '=' starts the token csv,
// so pre-override clients (chat.py) are parsed unchanged. Unknown
// keys are ignored (forward compat).
//
// The KV-cache / longest-common-prefix contract stays with the
// caller — this header only turns a line into a GenRequest.
// ═══════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace smoe::serve {

// One parsed GEN request. Sampling fields must be pre-loaded with the
// launch defaults by the caller; parse_gen overwrites only the fields
// the request explicitly sets.
struct GenRequest {
    uint32_t max_tokens  { 0 };
    float    temperature { 0.0f };
    float    top_p       { 0.0f };
    uint32_t top_k       { 0 };
    float    rep_penalty { 0.0f };
    uint32_t count       { 0 };      // parsed token ids in the caller's buffer
};

// Parse a "GEN ..." line (caller has already matched the prefix) into
// req + the caller's token buffer (capacity `cap`). Returns false on a
// malformed request or token overflow — the caller distinguishes the
// two by req.count == cap.
//
//   top_k_cap — hard ceiling for k= overrides, from the sampler's
//               pre-allocated scratch (Golden Rule: an override must
//               never reallocate mid-loop).
inline bool parse_gen(const std::string& line, GenRequest& req,
                      uint32_t* tokens, uint32_t cap, uint32_t top_k_cap)
{
    req.count = 0;
    size_t p = 4;   // past "GEN "
    size_t sp = line.find(' ', p);
    if (sp == std::string::npos) return false;

    req.max_tokens = static_cast<uint32_t>(std::atoi(line.c_str() + p));
    if (req.max_tokens == 0 || req.max_tokens > 8192) return false;
    p = sp + 1;

    // Optional sampling overrides: consume key=value fields until the
    // first field with no '=' (the token csv).
    while (p < line.size()) {
        size_t fend = line.find(' ', p);
        if (fend == std::string::npos) fend = line.size();
        size_t eq = line.find('=', p);
        if (eq == std::string::npos || eq >= fend) break;   // csv begins
        const char* val = line.c_str() + eq + 1;
        if (eq == p + 1) {
            switch (line[p]) {
                case 't':   // temperature (<1e-4 → greedy)
                    req.temperature = std::max(0.0f, std::strtof(val, nullptr));
                    break;
                case 'p':   // top_p, clamped to a sane range
                    req.top_p = std::min(std::max(std::strtof(val, nullptr), 0.01f), 1.0f);
                    break;
                case 'k': { // top_k, capped by the scratch pre-allocation
                    long kv = std::strtol(val, nullptr, 10);
                    const long kcap = static_cast<long>(top_k_cap);
                    req.top_k = static_cast<uint32_t>(std::min(std::max(kv, 1L), kcap));
                    break;
                }
                case 'r': { // repetition penalty (must be > 0)
                    float rv = std::strtof(val, nullptr);
                    if (rv > 0.0f) req.rep_penalty = rv;
                    break;
                }
                default: break;   // unknown key: ignore (forward compat)
            }
        }
        p = (fend < line.size()) ? fend + 1 : fend;
    }

    // Token csv.
    while (p < line.size()) {
        if (req.count >= cap) return false;   // overflow: count == cap
        char* endp = nullptr;
        unsigned long v = std::strtoul(line.c_str() + p, &endp, 10);
        if (endp == line.c_str() + p) return false;
        tokens[req.count++] = static_cast<uint32_t>(v);
        p = static_cast<size_t>(endp - line.c_str());
        if (p < line.size() && line[p] == ',') ++p;
    }
    return req.count > 0;
}

} // namespace smoe::serve
