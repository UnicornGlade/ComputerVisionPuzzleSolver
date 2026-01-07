#include "find_cycles.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <libbase/runtime_assert.h>

namespace find_cycles {

namespace {

constexpr float kEps = 1e-6f;

static bool is_better_cycle(float w, const std::vector<int>& cyc, float best_w, const std::vector<int>& best_cyc) {
    if (w > best_w + kEps) return true;
    if (std::fabs(w - best_w) <= kEps) {
        if (cyc.size() > best_cyc.size()) return true;
        if (cyc.size() == best_cyc.size()) return cyc < best_cyc; // lexicographic tie-break
    }
    return false;
}

struct DfsCtx {
    const graph::Graph* g = nullptr;
    const std::vector<char>* active = nullptr;

    int start = -1;

    std::vector<char> used;
    std::vector<int> path;
    float path_w = 0.0f;

    float best_w = -1.0f;
    std::vector<int> best_cycle;
};

static void dfs(DfsCtx& ctx, int v) {
    ctx.used[static_cast<std::size_t>(v)] = 1;
    ctx.path.push_back(v);

    const auto& edges = ctx.g->adj[static_cast<std::size_t>(v)];
    for (const auto& e : edges) {
        const int to = e.to;
        if (!(*ctx.active)[static_cast<std::size_t>(to)]) continue;

        // Canonical enumeration: ensure start is the minimum vertex in the cycle.
        // So we never visit vertices < start (except returning to start).
        if (to != ctx.start && to < ctx.start) continue;

        if (to == ctx.start) {
            // Found a directed cycle.
            // Allow self-loop cycle [start] (path size == 1, v == start, edge start->start).
            // Otherwise require at least 2 vertices in the cycle.
            if (ctx.path.size() == 1 && v != ctx.start) continue;
            if (ctx.path.size() >= 2 || (ctx.path.size() == 1 && v == ctx.start)) {
                const float cyc_w = ctx.path_w + e.w;
                const std::vector<int>& cyc = ctx.path; // no repeated start at end
                if (is_better_cycle(cyc_w, cyc, ctx.best_w, ctx.best_cycle)) {
                    ctx.best_w = cyc_w;
                    ctx.best_cycle = cyc;
                }
            }
            continue;
        }

        if (!ctx.used[static_cast<std::size_t>(to)]) {
            const float saved = ctx.path_w;
            ctx.path_w += e.w;
            dfs(ctx, to);
            ctx.path_w = saved;
        }
    }

    ctx.path.pop_back();
    ctx.used[static_cast<std::size_t>(v)] = 0;
}

static bool find_best_cycle_bruteforce(const graph::Graph& g,
                                       const std::vector<char>& active,
                                       std::vector<int>& out_cycle) {
    const int n = g.num_vertices();
    out_cycle.clear();

    float global_best_w = -1.0f;
    std::vector<int> global_best;

    for (int start = 0; start < n; ++start) {
        if (!active[static_cast<std::size_t>(start)]) continue;

        DfsCtx ctx;
        ctx.g = &g;
        ctx.active = &active;
        ctx.start = start;
        ctx.used.assign(static_cast<std::size_t>(n), 0);
        ctx.path.clear();
        ctx.path_w = 0.0f;
        ctx.best_w = -1.0f;
        ctx.best_cycle.clear();

        dfs(ctx, start);

        if (!ctx.best_cycle.empty()) {
            if (is_better_cycle(ctx.best_w, ctx.best_cycle, global_best_w, global_best)) {
                global_best_w = ctx.best_w;
                global_best = ctx.best_cycle;
            }
        }
    }

    if (global_best.empty()) return false;
    out_cycle = std::move(global_best);
    return true;
}

} // namespace

std::vector<std::vector<int>> find_cycles_greedy_bruteforce(const graph::Graph& g) {
    const int n = g.num_vertices();
    std::vector<char> active(static_cast<std::size_t>(n), 1);

    std::vector<std::vector<int>> cycles;
    cycles.reserve(static_cast<std::size_t>(n));

    while (true) {
        std::vector<int> best;
        if (!find_best_cycle_bruteforce(g, active, best)) break;

        // Remove vertices of the found cycle
        for (int v : best) {
            rassert(v >= 0 && v < n, "internal: bad vertex in cycle", v, n);
            active[static_cast<std::size_t>(v)] = 0;
        }

        cycles.push_back(std::move(best));
    }

    return cycles;
}

} // namespace find_cycles
