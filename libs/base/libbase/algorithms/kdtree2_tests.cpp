#include "kdtree2.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using kdtree2::KDTree2;
using kdtree2::Neighbor;
using kdtree2::SearchParams;

static float dist2(const point2f& a, const point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

static std::vector<Neighbor> brute_knn(const std::vector<point2f>& pts, const point2f& q, int k) {
    std::vector<std::pair<float, int>> tmp;
    tmp.reserve(pts.size());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) tmp.push_back({dist2(pts[i], q), i});
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a.first < b.first; });

    k = std::min(k, static_cast<int>(pts.size()));
    std::vector<Neighbor> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) out.push_back({tmp[i].second, std::sqrt(tmp[i].first)});
    return out;
}

static std::vector<Neighbor> brute_radius(const std::vector<point2f>& pts, const point2f& q, float r) {
    const float r2 = r * r;
    std::vector<std::pair<float, int>> tmp;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const float d2 = dist2(pts[i], q);
        if (d2 <= r2) tmp.push_back({d2, i});
    }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a.first < b.first; });

    std::vector<Neighbor> out;
    out.reserve(tmp.size());
    for (auto& [d2, idx] : tmp) out.push_back({idx, std::sqrt(d2)});
    return out;
}

TEST(kdtree2, KNNExactWhenUnlimited) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-100.0f, 100.0f);

    std::vector<point2f> pts;
    pts.reserve(2000);
    for (int i = 0; i < 2000; ++i) pts.push_back(point2f{U(rng), U(rng)});

    KDTree2 tree(pts);

    SearchParams sp;
    sp.max_nodes_visited = 1'000'000; // effectively unlimited
    sp.eps = 0.0f;

    const point2f q{1.25f, -3.5f};

    const auto kd = tree.knn_search(q, 10, sp);
    const auto br = brute_knn(pts, q, 10);

    ASSERT_EQ(kd.size(), br.size());
    for (std::size_t i = 0; i < kd.size(); ++i) {
        EXPECT_EQ(kd[i].first, br[i].first);
        EXPECT_NEAR(kd[i].second, br[i].second, 1e-5f);
        if (i + 1 < kd.size()) EXPECT_LE(kd[i].second, kd[i + 1].second);
    }
}

TEST(kdtree2, RadiusExactWhenUnlimited) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-50.0f, 50.0f);

    std::vector<point2f> pts;
    pts.reserve(1500);
    for (int i = 0; i < 1500; ++i) pts.push_back(point2f{U(rng), U(rng)});

    KDTree2 tree(pts);

    SearchParams sp;
    sp.max_nodes_visited = 1'000'000;
    sp.eps = 0.0f;

    const point2f q{0.0f, 0.0f};
    const float r = 10.0f;

    const auto kd = tree.radius_search(q, r, sp);
    const auto br = brute_radius(pts, q, r);

    ASSERT_EQ(kd.size(), br.size());
    for (std::size_t i = 0; i < kd.size(); ++i) {
        EXPECT_EQ(kd[i].first, br[i].first);
        EXPECT_NEAR(kd[i].second, br[i].second, 1e-5f);
        if (i + 1 < kd.size()) EXPECT_LE(kd[i].second, kd[i + 1].second);
    }
}

TEST(kdtree2, ApproximateStillSortedAndSized) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-100.0f, 100.0f);

    std::vector<point2f> pts;
    pts.reserve(5000);
    for (int i = 0; i < 5000; ++i) pts.push_back(point2f{U(rng), U(rng)});

    KDTree2 tree(pts);

    SearchParams sp;
    sp.max_nodes_visited = 64; // very approximate
    sp.eps = 0.2f;

    const point2f q{10.0f, -20.0f};
    const auto kd = tree.knn_search(q, 20, sp);

    ASSERT_LE(kd.size(), 20u);
    for (std::size_t i = 1; i < kd.size(); ++i) {
        EXPECT_LE(kd[i - 1].second, kd[i].second);
    }
}

TEST(kdtree2, EmptyInput) {
    KDTree2 tree;
    tree.build({});
    EXPECT_EQ(tree.size(), 0);

    SearchParams sp;
    sp.max_nodes_visited = 1000;

    const point2f q{0.0f, 0.0f};
    EXPECT_TRUE(tree.knn_search(q, 5, sp).empty());
    EXPECT_TRUE(tree.radius_search(q, 1.0f, sp).empty());
}
