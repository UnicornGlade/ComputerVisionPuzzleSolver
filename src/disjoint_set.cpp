#include "disjoint_set.h"

#include <algorithm>

DisjointSetUnion::DisjointSetUnion(std::size_t n) : parent_(n), sz_(n, 1) {
    for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
}

std::size_t DisjointSetUnion::find(std::size_t x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}

std::size_t DisjointSetUnion::find(std::size_t x) const {
    while (parent_[x] != x) x = parent_[x];
    return x;
}

std::pair<std::size_t, std::size_t> DisjointSetUnion::unite_roots(std::size_t ra, std::size_t rb) {
    if (ra == rb) return {ra, ra};

    if (sz_[ra] < sz_[rb]) std::swap(ra, rb);
    parent_[rb] = ra;
    sz_[ra] += sz_[rb];
    return {ra, rb};
}

bool DisjointSetUnion::unite(std::size_t a, std::size_t b) {
    const std::size_t ra = find(a);
    const std::size_t rb = find(b);
    const auto [kept, absorbed] = unite_roots(ra, rb);
    return kept != absorbed;
}

std::size_t DisjointSetUnion::set_size(std::size_t x) const {
    const std::size_t r = find(x);
    return sz_[r];
}
