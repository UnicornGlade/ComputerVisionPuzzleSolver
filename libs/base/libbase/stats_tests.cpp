#include "stats.h"
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

TEST(Stats, MinMax_Int) {
    std::vector<int> v{3, -2, 7, 1};
    EXPECT_EQ(stats::minValue(v), -2);
    EXPECT_EQ(stats::maxValue(v), 7);
}

TEST(Stats, MinMax_SizeT) {
    std::vector<std::size_t> v{10u, 2u, 7u, 7u};
    EXPECT_EQ(stats::minValue(v), 2u);
    EXPECT_EQ(stats::maxValue(v), 10u);
}

TEST(Stats, MinMax_Uint8) {
    std::vector<std::uint8_t> v{std::uint8_t{200}, std::uint8_t{3}, std::uint8_t{255}};
    EXPECT_EQ(static_cast<int>(stats::minValue(v)), 3);
    EXPECT_EQ(static_cast<int>(stats::maxValue(v)), 255);
}

TEST(Stats, MedianEven_Int) {
    std::vector<int> v{0, 10, 20, 30};
    EXPECT_DOUBLE_EQ(stats::median(v), 15.0);
}

TEST(Stats, PercentileLinear_Int) {
    std::vector<int> v{0, 10, 20, 30};
    EXPECT_DOUBLE_EQ(stats::percentile(v, 0), 0);
    EXPECT_DOUBLE_EQ(stats::percentile(v, 100), 30);
    EXPECT_DOUBLE_EQ(stats::percentile(v, 25), 7.5);
    EXPECT_DOUBLE_EQ(stats::percentile(v, 75), 22.5);
}

TEST(Stats, PercentileRangeChecks) {
    std::vector<double> v{1, 2, 3};
    EXPECT_THROW(stats::percentile(v, -1), std::invalid_argument);
    EXPECT_THROW(stats::percentile(v, 101), std::invalid_argument);
}

TEST(Stats, EmptyThrowsForNumbers) {
    std::vector<int> v;
    EXPECT_THROW(stats::minValue(v), std::invalid_argument);
    EXPECT_THROW(stats::maxValue(v), std::invalid_argument);
    EXPECT_THROW(stats::median(v), std::invalid_argument);
    EXPECT_THROW(stats::percentile(v, 50), std::invalid_argument);
}

TEST(Stats, PreviewSmall_Uint8PrintedAsNumber) {
    std::vector<std::uint8_t> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(stats::previewValues(v), "10 values - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
}

TEST(Stats, PreviewLarge_Int) {
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(stats::previewValues(v), "12 values - [1, 2, 3, 4, 5, ... 8, 9, 10, 11, 12]");
}

TEST(Stats, Summary_Int) {
    // 1..20
    std::vector<int> v;
    for (int i = 1; i <= 20; ++i)
        v.push_back(i);

    EXPECT_EQ(stats::summaryStats(v), "20 values - (min=1 10%=2.9 median=10.5 90%=18.1 max=20)");
}

TEST(Stats, Summary_FloatDefaultDecimals2) {
    // 1..20 as float
    std::vector<float> v;
    for (int i = 1; i <= 20; ++i)
        v.push_back(static_cast<float>(i));

    EXPECT_EQ(stats::summaryStats(v), "20 values - (min=1.00 10%=2.90 median=10.50 90%=18.10 max=20.00)");
}

TEST(Stats, Summary_DoubleCustomDecimals3) {
    std::vector<double> v{0.1, 0.2, 0.3, 0.4};
    // median = 0.25, 10% pos=0.3 => 0.13, 90% pos=2.7 => 0.37
    EXPECT_EQ(stats::summaryStats(v, 3), "4 values - (min=0.100 10%=0.130 median=0.250 90%=0.370 max=0.400)");
}

TEST(Stats, SummaryEmpty) {
    std::vector<double> v;
    EXPECT_EQ(stats::summaryStats(v), "0 values - (empty)");
    EXPECT_EQ(stats::summaryStats(v, 5), "0 values - (empty)");
}
