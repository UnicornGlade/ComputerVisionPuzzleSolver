#pragma once

#include <vector>

#include <libbase/graph.h>

namespace find_cycles {

// Greedy decomposition into vertex-disjoint cycles:
// 1) Find maximum-total-weight simple directed cycle by brute force
// 2) Remove its vertices
// 3) Repeat until no cycle exists
//
// Returns cycles as vectors of vertex indices (start is minimal vertex in that cycle).
std::vector<std::vector<int>> find_cycles_greedy_bruteforce(const graph::Graph& g);

} // namespace find_cycles
