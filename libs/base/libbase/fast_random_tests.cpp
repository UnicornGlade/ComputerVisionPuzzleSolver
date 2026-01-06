#include "fast_random.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

TEST(FastRandom, DeterministicSameSeed) {
    FastRandom a(123u);
    FastRandom b(123u);

    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(a.nextU32(), b.nextU32());
    }
}

TEST(FastRandom, DifferentSeedsDifferentSequenceEarly) {
    FastRandom a(123u);
    FastRandom b(124u);

    // Not a hard guarantee for RNGs, but for sane mixSeed+xorshift it should hold.
    bool anyDiff = false;
    for (int i = 0; i < 32; ++i) {
        if (a.nextU32() != b.nextU32()) {
            anyDiff = true;
            break;
        }
    }
    EXPECT_TRUE(anyDiff);
}

TEST(FastRandom, NextU32NotAllSame) {
    FastRandom r(1u);

    uint32_t first = r.nextU32();
    bool anyDiff = false;
    for (int i = 0; i < 1000; ++i) {
        if (r.nextU32() != first) {
            anyDiff = true;
            break;
        }
    }
    EXPECT_TRUE(anyDiff);
}

TEST(FastRandom, NextIntCoversSignedRangeType) {
    FastRandom r(777u);

    // Just verify type/convertibility and variability.
    int32_t first = r.nextInt();
    bool anyDiff = false;
    for (int i = 0; i < 1000; ++i) {
        if (r.nextInt() != first) {
            anyDiff = true;
            break;
        }
    }
    EXPECT_TRUE(anyDiff);
}

TEST(FastRandom, NextIntRangeInclusiveBasic) {
    FastRandom r(42u);

    for (int i = 0; i < 10000; ++i) {
        int32_t x = r.nextInt(-3, 3);
        EXPECT_GE(x, -3);
        EXPECT_LE(x, 3);
    }

    // Degenerate range should always return that value.
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(r.nextInt(5, 5), 5);
    }
}

TEST(FastRandom, NextIntRangeSwappedBounds) {
    FastRandom r(42u);

    for (int i = 0; i < 10000; ++i) {
        int32_t x = r.nextInt(10, -10);
        EXPECT_GE(x, -10);
        EXPECT_LE(x, 10);
    }
}

TEST(FastRandom, NextIntRangeFullInt32DoesNotCrash) {
    FastRandom r(999u);

    // This covers the corner case when range computation may overflow to 0.
    for (int i = 0; i < 1000; ++i) {
        int32_t x = r.nextInt(std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::max());
        (void)x;
    }
    SUCCEED();
}

TEST(FastRandom, NextFloatUnitInterval) {
    FastRandom r(314159u);

    for (int i = 0; i < 200000; ++i) {
        float x = r.nextFloat();
        EXPECT_GE(x, 0.0f);
        EXPECT_LT(x, 1.0f);
    }
}

TEST(FastRandom, NextFloatRangeBasic) {
    FastRandom r(2718u);

    for (int i = 0; i < 200000; ++i) {
        float x = r.nextFloat(-2.5f, 3.5f);
        EXPECT_GE(x, -2.5f);
        EXPECT_LT(x, 3.5f); // upper bound exclusive by contract
    }

    // Degenerate range should always return that value.
    for (int i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(r.nextFloat(1.25f, 1.25f), 1.25f);
    }
}

TEST(FastRandom, NextFloatRangeSwappedBounds) {
    FastRandom r(2718u);

    for (int i = 0; i < 200000; ++i) {
        float x = r.nextFloat(10.0f, -10.0f);
        EXPECT_GE(x, -10.0f);
        EXPECT_LT(x, 10.0f);
    }
}

TEST(FastRandom, BasicUniquenessSmoke) {
    FastRandom r(123u);

    // Not a distribution test: just ensure we don't get stuck on a tiny cycle immediately.
    std::unordered_set<uint32_t> s;
    s.reserve(4096);

    for (int i = 0; i < 4096; ++i) {
        s.insert(r.nextU32());
    }

    // Expect many unique values (extremely likely unless broken).
    EXPECT_GT(s.size(), 4000u);
}
