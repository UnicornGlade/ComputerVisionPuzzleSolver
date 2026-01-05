#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <source_location>

class DisjointSetUnion final {
public:
    explicit DisjointSetUnion(std::size_t n);

    std::size_t size() const noexcept { return parent_.size(); }

    std::size_t find(std::size_t x, std::source_location loc = std::source_location::current());
    std::size_t find(std::size_t x, std::source_location loc = std::source_location::current()) const; // no path compression

    // Unites sets containing a and b. Returns true if merged.
    bool unite(std::size_t a, std::size_t b, std::source_location loc = std::source_location::current());

    // Like unite(), but expects roots and returns {root_kept, root_absorbed} if merged.
    // If not merged (already same root) returns {root, root}.
    std::pair<std::size_t, std::size_t> unite_roots(std::size_t ra, std::size_t rb, std::source_location loc = std::source_location::current());

    std::size_t set_size(std::size_t x, std::source_location loc = std::source_location::current()) const;

private:

    std::vector<std::size_t> parent_;
    std::vector<std::size_t> sz_;
};
