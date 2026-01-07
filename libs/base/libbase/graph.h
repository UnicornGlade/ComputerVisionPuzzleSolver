#pragma once

#include <cstddef>
#include <vector>

#include <libbase/runtime_assert.h>

namespace graph {

struct Edge {
    int to = -1;
    float w = 0.0f; // w >= 0
};

struct Graph {
    std::vector<std::vector<Edge>> adj;

    Graph() = default;
    explicit Graph(int n) : adj(static_cast<std::size_t>(n)) { rassert(n >= 0, "Graph: n must be >= 0", n); }

    int num_vertices() const noexcept { return static_cast<int>(adj.size()); }

    void add_edge(int from, int to, float w) {
        rassert(from >= 0 && from < num_vertices(), "add_edge: bad from", from, num_vertices());
        rassert(to >= 0 && to < num_vertices(), "add_edge: bad to", to, num_vertices());
        rassert(w >= 0.0f, "add_edge: weight must be >= 0", w);
        adj[static_cast<std::size_t>(from)].push_back(Edge{to, w});
    }
};

} // namespace graph
