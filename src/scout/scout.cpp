// ═══════════════════════════════════════════════════════════════
// scout.cpp — S-MoE Engine · Surface Scout Implementation
// ═══════════════════════════════════════════════════════════════
// Phase 4 — Week 4 implementation target (heuristic routing).
// Phase 5+ — distilled neural routing head.
// ═══════════════════════════════════════════════════════════════

#include "scout.hpp"

// TODO — Week 4

namespace smoe::scout {

struct Scout::Impl {};

Scout::Scout(const char* path) { (void)path; /* TODO */ }
Scout::~Scout() = default;

ScoutOutput Scout::forward(uint32_t token_id) {
    (void)token_id;
    return {}; // TODO
}

void Scout::reset_context() {} // TODO

} // namespace smoe::scout
