#include "disjoint_set.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// ------------------------- Helpers -------------------------

static bool containsRootPair(std::pair<std::size_t, std::size_t> p, std::size_t a, std::size_t b) {
    return (p.first == a && p.second == b) || (p.first == b && p.second == a);
}

// A very simple (but independent) DSU reference for stress tests.
class RefDsu {
  public:
    explicit RefDsu(std::size_t n) : parent_(n), size_(n, 1) {
        for (std::size_t i = 0; i < n; ++i)
            parent_[i] = i;
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    bool unite(std::size_t a, std::size_t b) {
        auto ra = find(a);
        auto rb = find(b);
        if (ra == rb)
            return false;
        if (size_[ra] < size_[rb])
            std::swap(ra, rb);
        parent_[rb] = ra;
        size_[ra] += size_[rb];
        return true;
    }

    std::size_t set_size(std::size_t x) { return size_[find(x)]; }

  private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};

// Naive reference (O(N) union) for small randomized test; good for catching subtle size bugs.
class NaiveComponents {
  public:
    explicit NaiveComponents(std::size_t n) : comp_(n) {
        for (std::size_t i = 0; i < n; ++i)
            comp_[i] = static_cast<int>(i);
    }

    bool unite(std::size_t a, std::size_t b) {
        int ca = comp_[a];
        int cb = comp_[b];
        if (ca == cb)
            return false;
        // Recolor all nodes of cb into ca
        for (auto &c : comp_) {
            if (c == cb)
                c = ca;
        }
        return true;
    }

    bool same(std::size_t a, std::size_t b) const { return comp_[a] == comp_[b]; }

    std::size_t set_size(std::size_t x) const {
        int cx = comp_[x];
        std::size_t cnt = 0;
        for (int c : comp_)
            if (c == cx)
                ++cnt;
        return cnt;
    }

  private:
    std::vector<int> comp_;
};

// ------------------------- Unit tests -------------------------

TEST(DisjointSetUnion, ConstructorInitialState) {
    DisjointSetUnion dsu(5);

    EXPECT_EQ(dsu.size(), 5u);

    for (std::size_t i = 0; i < dsu.size(); ++i) {
        EXPECT_EQ(dsu.find(i), i);
        EXPECT_EQ(dsu.set_size(i), 1u);
    }

    const DisjointSetUnion &cdsu = dsu;
    for (std::size_t i = 0; i < cdsu.size(); ++i) {
        EXPECT_EQ(cdsu.find(i), i);
    }
}

TEST(DisjointSetUnion, UniteMergesAndSizes) {
    DisjointSetUnion dsu(6);

    EXPECT_TRUE(dsu.unite(0, 1));
    EXPECT_TRUE(dsu.unite(1, 2));

    auto r0 = dsu.find(0);
    EXPECT_EQ(dsu.find(1), r0);
    EXPECT_EQ(dsu.find(2), r0);
    EXPECT_EQ(dsu.set_size(0), 3u);
    EXPECT_EQ(dsu.set_size(1), 3u);
    EXPECT_EQ(dsu.set_size(2), 3u);

    // Disjoint singleton stays singleton
    EXPECT_EQ(dsu.set_size(3), 1u);

    // Already connected => false
    EXPECT_FALSE(dsu.unite(0, 2));
    EXPECT_FALSE(dsu.unite(2, 0));

    // After merging a singleton in, size grows
    EXPECT_TRUE(dsu.unite(2, 3));
    EXPECT_EQ(dsu.set_size(0), 4u);
    EXPECT_EQ(dsu.set_size(3), 4u);
}

TEST(DisjointSetUnion, FindIdempotent) {
    DisjointSetUnion dsu(10);

    dsu.unite(1, 2);
    dsu.unite(2, 3);
    dsu.unite(5, 6);

    for (std::size_t i = 0; i < dsu.size(); ++i) {
        auto r1 = dsu.find(i);
        auto r2 = dsu.find(r1);
        EXPECT_EQ(r1, r2); // root's root is itself
    }
}

TEST(DisjointSetUnion, UniteRootsReturnsPairAndKeepsInvariants) {
    DisjointSetUnion dsu(8);

    // Build two components: {0,1,2} and {4,5}
    ASSERT_TRUE(dsu.unite(0, 1));
    ASSERT_TRUE(dsu.unite(1, 2));
    ASSERT_TRUE(dsu.unite(4, 5));

    std::size_t rA = dsu.find(0);
    std::size_t rB = dsu.find(4);
    ASSERT_NE(rA, rB);

    const std::size_t sA = dsu.set_size(rA);
    const std::size_t sB = dsu.set_size(rB);

    auto merged = dsu.unite_roots(rA, rB);

    // Returned roots must be exactly the two input roots (in any order).
    EXPECT_TRUE(containsRootPair(merged, rA, rB));

    // After merge, both belong to same root (which is merged.first by contract).
    std::size_t rootKept = merged.first;
    std::size_t rootAbsorbed = merged.second;

    EXPECT_EQ(dsu.find(rootKept), rootKept);
    EXPECT_EQ(dsu.find(rootAbsorbed), rootKept);

    // Component size must be sum.
    EXPECT_EQ(dsu.set_size(rootKept), sA + sB);

    // Re-merging same root => {root, root}
    auto again = dsu.unite_roots(rootKept, rootKept);
    EXPECT_EQ(again.first, rootKept);
    EXPECT_EQ(again.second, rootKept);
}

TEST(DisjointSetUnion, ConstFindMatchesNonConstFind) {
    DisjointSetUnion dsu(7);
    dsu.unite(0, 1);
    dsu.unite(1, 2);
    dsu.unite(3, 4);

    const DisjointSetUnion &cdsu = dsu;

    for (std::size_t i = 0; i < dsu.size(); ++i) {
        EXPECT_EQ(dsu.find(i), cdsu.find(i));
    }
}

TEST(DisjointSetUnion, RandomizedSmallAgainstNaive) {
    constexpr std::size_t N = 2000;
    constexpr int OPS = 20000;

    DisjointSetUnion dsu(N);
    NaiveComponents ref(N);

    std::mt19937 rng(12345u);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(N - 1));
    std::uniform_int_distribution<int> coin(0, 99);

    for (int op = 0; op < OPS; ++op) {
        std::size_t a = static_cast<std::size_t>(dist(rng));
        std::size_t b = static_cast<std::size_t>(dist(rng));

        if (coin(rng) < 70) {
            // Mostly unions
            bool r1 = dsu.unite(a, b);
            bool r2 = ref.unite(a, b);
            ASSERT_EQ(r1, r2);
        } else {
            // Queries: connectivity + set sizes
            bool same1 = (dsu.find(a) == dsu.find(b));
            bool same2 = ref.same(a, b);
            ASSERT_EQ(same1, same2);

            ASSERT_EQ(dsu.set_size(a), ref.set_size(a));
            ASSERT_EQ(dsu.set_size(b), ref.set_size(b));
        }
    }

    // Final spot-check for all elements: sizes must match naive.
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(dsu.set_size(i), ref.set_size(i));
    }
}

TEST(DisjointSetUnion, StressLargeAgainstRefDsu) {
    constexpr std::size_t N = 100'000;
    constexpr int OPS = 300'000;

    DisjointSetUnion dsu(N);
    RefDsu ref(N);

    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<std::uint32_t> dist(0, static_cast<std::uint32_t>(N - 1));
    std::uniform_int_distribution<int> coin(0, 99);

    for (int op = 0; op < OPS; ++op) {
        std::size_t a = dist(rng);
        std::size_t b = dist(rng);

        if (coin(rng) < 80) {
            // Unite
            bool r1 = dsu.unite(a, b);
            bool r2 = ref.unite(a, b);
            ASSERT_EQ(r1, r2);
        } else {
            // Query
            bool same1 = (dsu.find(a) == dsu.find(b));
            bool same2 = (ref.find(a) == ref.find(b));
            ASSERT_EQ(same1, same2);
        }
    }

    // Verify random sample of roots and sizes.
    for (int i = 0; i < 2000; ++i) {
        std::size_t x = dist(rng);
        std::size_t rx1 = dsu.find(x);
        std::size_t rx2 = ref.find(x);

        // Not necessarily same numeric root id, but must represent same partition.
        // So compare by checking connectivity against a second random element.
        std::size_t y = dist(rng);
        bool same1 = (dsu.find(x) == dsu.find(y));
        bool same2 = (ref.find(x) == ref.find(y));
        ASSERT_EQ(same1, same2);

        // Size must match.
        ASSERT_EQ(dsu.set_size(x), ref.set_size(x));

        // Root should be stable under find
        ASSERT_EQ(dsu.find(rx1), rx1);

        (void)rx2;
    }
}
