#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "find_cycles.h"

static float cycle_weight(const graph::Graph& g, const std::vector<int>& cyc) {
    // Sum weights along cyc[0]->cyc[1]->...->cyc[last]->cyc[0]
    rassert(!cyc.empty(), 1234512321321);

    float sum = 0.0f;
    for (std::size_t i = 0; i < cyc.size(); ++i) {
        const int u = cyc[i];
        const int v = cyc[(i + 1) % cyc.size()];

        bool found = false;
        for (const auto& e : g.adj[static_cast<std::size_t>(u)]) {
            if (e.to == v) {
                sum += e.w;
                found = true;
                break;
            }
        }
        rassert(found, "Edge missing in cycle: " + std::to_string(u) + " -> " + std::to_string(v));
    }
    return sum;
}

static bool all_vertices_disjoint(const std::vector<std::vector<int>>& cycles) {
    std::vector<int> all;
    for (const auto& c : cycles) all.insert(all.end(), c.begin(), c.end());
    std::sort(all.begin(), all.end());
    return std::adjacent_find(all.begin(), all.end()) == all.end();
}

TEST(find_cycles, TwoDisjointCycles_PicksHeavierFirst) {
    graph::Graph g(6);

    // Cycle A: 0->1->2->0 total 12
    g.add_edge(0, 1, 1.0f);
    g.add_edge(1, 2, 1.0f);
    g.add_edge(2, 0, 10.0f);

    // Cycle B: 3->4->5->3 total 6
    g.add_edge(3, 4, 2.0f);
    g.add_edge(4, 5, 2.0f);
    g.add_edge(5, 3, 2.0f);

    // Extra edges (noise)
    g.add_edge(0, 3, 0.1f);
    g.add_edge(4, 0, 0.1f);

    const auto cycles = find_cycles::find_cycles_greedy_bruteforce(g);
    ASSERT_EQ(cycles.size(), 2u);
    EXPECT_TRUE(all_vertices_disjoint(cycles));

    // Heavier first
    const float w0 = cycle_weight(g, cycles[0]);
    const float w1 = cycle_weight(g, cycles[1]);
    EXPECT_GT(w0, w1);

    // Canonical start: minimal vertex in cycle.
    EXPECT_EQ(*std::min_element(cycles[0].begin(), cycles[0].end()), cycles[0][0]);
    EXPECT_EQ(*std::min_element(cycles[1].begin(), cycles[1].end()), cycles[1][0]);
}

TEST(find_cycles, OverlappingCycles_RemovesVertices) {
    graph::Graph g(5);

    // Cycle A: 0->1->2->0 total 11
    g.add_edge(0, 1, 5.0f);
    g.add_edge(1, 2, 5.0f);
    g.add_edge(2, 0, 1.0f);

    // Cycle B overlaps at vertex 2: 2->3->4->2 total 10
    g.add_edge(2, 3, 4.0f);
    g.add_edge(3, 4, 3.0f);
    g.add_edge(4, 2, 3.0f);

    const auto cycles = find_cycles::find_cycles_greedy_bruteforce(g);
    ASSERT_EQ(cycles.size(), 1u);
    EXPECT_TRUE(all_vertices_disjoint(cycles));
    EXPECT_EQ(cycles[0][0], 0); // minimal vertex is 0
}

TEST(find_cycles, SelfLoopHandled) {
    graph::Graph g(3);

    // Self-loop at 0
    g.add_edge(0, 0, 5.0f);

    // 2-cycle 1<->2 total 6
    g.add_edge(1, 2, 3.0f);
    g.add_edge(2, 1, 3.0f);

    const auto cycles = find_cycles::find_cycles_greedy_bruteforce(g);
    ASSERT_EQ(cycles.size(), 2u);
    EXPECT_TRUE(all_vertices_disjoint(cycles));

    const float w0 = cycle_weight(g, cycles[0]);
    const float w1 = cycle_weight(g, cycles[1]);
    EXPECT_GE(w0, w1);

    // One of cycles should be [0]
    bool has0 = false;
    for (const auto& c : cycles) if (c.size() == 1 && c[0] == 0) has0 = true;
    EXPECT_TRUE(has0);
}

TEST(find_cycles, NoCycles) {
    graph::Graph g(4);
    g.add_edge(0, 1, 1.0f);
    g.add_edge(1, 2, 1.0f);
    g.add_edge(2, 3, 1.0f);

    const auto cycles = find_cycles::find_cycles_greedy_bruteforce(g);
    EXPECT_TRUE(cycles.empty());
}
