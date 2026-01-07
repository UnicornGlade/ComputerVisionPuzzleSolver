#include "utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(utils, FilterVector_Evens) {
    const std::vector<int> v = {1, 2, 3, 4, 5, 6};

    const auto evens = filter_vector(v, [](int x) { return (x % 2) == 0; });

    ASSERT_EQ(evens.size(), 3u);
    EXPECT_EQ(evens[0], 2);
    EXPECT_EQ(evens[1], 4);
    EXPECT_EQ(evens[2], 6);
}

TEST(utils, FilterVector_Empty) {
    const std::vector<int> v = {1, 3, 5};

    const auto evens = filter_vector(v, [](int x) { return (x % 2) == 0; });

    EXPECT_TRUE(evens.empty());
}

TEST(utils, MapVector_IntToString) {
    const std::vector<int> v = {7, 8, 9};

    const auto s = map_vector(v, [](int x) { return std::to_string(x); });

    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(s[0], "7");
    EXPECT_EQ(s[1], "8");
    EXPECT_EQ(s[2], "9");
}

TEST(utils, MapVector_StructToField) {
    struct P {
        int x = 0;
        int y = 0;
    };

    const std::vector<P> v = {{1, 2}, {3, 4}, {5, 6}};

    const auto xs = map_vector(v, [](const P& p) { return p.x; });

    ASSERT_EQ(xs.size(), 3u);
    EXPECT_EQ(xs[0], 1);
    EXPECT_EQ(xs[1], 3);
    EXPECT_EQ(xs[2], 5);
}
