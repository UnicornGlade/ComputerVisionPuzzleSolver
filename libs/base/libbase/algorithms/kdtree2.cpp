#include "kdtree2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <libbase/runtime_assert.h>

namespace kdtree2 {

KDTree2::KDTree2(std::vector<point2f> points) { build(std::move(points)); }

void KDTree2::build(std::vector<point2f> points) {
    points_ = std::move(points);
    indices_.resize(points_.size());
    for (std::size_t i = 0; i < points_.size(); ++i) indices_[i] = static_cast<int>(i);

    nodes_.clear();
    nodes_.reserve(points_.size());
    root_ = points_.empty() ? -1 : build_rec_(0, static_cast<int>(points_.size()));
}

int KDTree2::size() const noexcept { return static_cast<int>(points_.size()); }

int KDTree2::build_rec_(int l, int r) {
    if (l >= r) return -1;

    // Choose axis by spread.
    float minx = std::numeric_limits<float>::infinity();
    float miny = std::numeric_limits<float>::infinity();
    float maxx = -std::numeric_limits<float>::infinity();
    float maxy = -std::numeric_limits<float>::infinity();

    for (int i = l; i < r; ++i) {
        const point2f& p = points_[static_cast<std::size_t>(indices_[i])];
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }

    const float spanx = maxx - minx;
    const float spany = maxy - miny;
    const int axis = (spanx >= spany) ? 0 : 1;

    const int m = (l + r) / 2;

    // Median split by chosen axis.
    std::nth_element(indices_.begin() + l, indices_.begin() + m, indices_.begin() + r,
                     [&](int ia, int ib) {
                         const point2f& a = points_[static_cast<std::size_t>(ia)];
                         const point2f& b = points_[static_cast<std::size_t>(ib)];
                         return (axis == 0) ? (a.x < b.x) : (a.y < b.y);
                     });

    const int node_id = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    Node& n = nodes_.back();
    n.axis = axis;
    n.idx = indices_[m];

    n.left = build_rec_(l, m);
    n.right = build_rec_(m + 1, r);

    // Init bbox from point itself; then merge children.
    const point2f& p = points_[static_cast<std::size_t>(n.idx)];
    n.minx = n.maxx = p.x;
    n.miny = n.maxy = p.y;

    update_bbox_from_children_(node_id);
    return node_id;
}

void KDTree2::update_bbox_from_children_(int node_id) {
    Node& n = nodes_[static_cast<std::size_t>(node_id)];

    auto merge_child = [&](int child_id) {
        if (child_id < 0) return;
        const Node& c = nodes_[static_cast<std::size_t>(child_id)];
        n.minx = std::min(n.minx, c.minx);
        n.miny = std::min(n.miny, c.miny);
        n.maxx = std::max(n.maxx, c.maxx);
        n.maxy = std::max(n.maxy, c.maxy);
    };

    merge_child(n.left);
    merge_child(n.right);
}

float KDTree2::dist2_point_(const point2f& a, const point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

float KDTree2::dist2_bbox_(const Node& n, const point2f& q) {
    // Squared distance from point to axis-aligned bbox (0 if inside).
    float dx = 0.0f;
    if (q.x < n.minx) dx = n.minx - q.x;
    else if (q.x > n.maxx) dx = q.x - n.maxx;

    float dy = 0.0f;
    if (q.y < n.miny) dy = n.miny - q.y;
    else if (q.y > n.maxy) dy = q.y - n.maxy;

    return dx * dx + dy * dy;
}

std::vector<Neighbor> KDTree2::knn_search(const point2f& q, int k, const SearchParams& params) const {
    rassert(k >= 0, "k must be >= 0", k);
    if (k == 0 || points_.empty() || root_ < 0) return {};

    k = std::min(k, static_cast<int>(points_.size()));

    const int max_visits = (params.max_nodes_visited <= 0) ? std::numeric_limits<int>::max() : params.max_nodes_visited;
    const float eps = std::max(0.0f, params.eps);
    const float eps_factor = (1.0f + eps) * (1.0f + eps);

    struct QItem {
        float d2 = 0.0f;
        int node = -1;
    };
    struct QCmp {
        bool operator()(const QItem& a, const QItem& b) const { return a.d2 > b.d2; } // min-heap
    };

    std::priority_queue<QItem, std::vector<QItem>, QCmp> pq;
    pq.push(QItem{dist2_bbox_(nodes_[static_cast<std::size_t>(root_)], q), root_});

    // Best neighbors: max-heap by distance (top is worst).
    using BestItem = std::pair<float, int>; // (dist2, index)
    std::priority_queue<BestItem> best;

    int visited = 0;

    auto consider_point = [&](int idx) {
        const float d2 = dist2_point_(points_[static_cast<std::size_t>(idx)], q);
        if (static_cast<int>(best.size()) < k) {
            best.push({d2, idx});
        } else if (d2 < best.top().first) {
            best.pop();
            best.push({d2, idx});
        }
    };

    while (!pq.empty()) {
        const QItem cur = pq.top();
        pq.pop();

        // Stop if exact/eps condition allows it.
        if (static_cast<int>(best.size()) == k) {
            const float worst = best.top().first;
            if (cur.d2 > worst * eps_factor) break;
        }

        if (visited >= max_visits && static_cast<int>(best.size()) == k) break;
        ++visited;

        const Node& n = nodes_[static_cast<std::size_t>(cur.node)];
        consider_point(n.idx);

        if (n.left >= 0) {
            pq.push(QItem{dist2_bbox_(nodes_[static_cast<std::size_t>(n.left)], q), n.left});
        }
        if (n.right >= 0) {
            pq.push(QItem{dist2_bbox_(nodes_[static_cast<std::size_t>(n.right)], q), n.right});
        }
    }

    std::vector<Neighbor> out;
    out.reserve(best.size());
    while (!best.empty()) {
        const auto [d2, idx] = best.top();
        best.pop();
        out.push_back({idx, std::sqrt(d2)});
    }
    std::sort(out.begin(), out.end(), [](const Neighbor& a, const Neighbor& b) { return a.second < b.second; });
    return out;
}

std::vector<Neighbor> KDTree2::radius_search(const point2f& q, float radius, const SearchParams& params) const {
    rassert(radius >= 0.0f, "radius must be >= 0", radius);
    if (points_.empty() || root_ < 0) return {};

    const float r2 = radius * radius;
    const int max_visits = (params.max_nodes_visited <= 0) ? std::numeric_limits<int>::max() : params.max_nodes_visited;

    struct QItem {
        float d2 = 0.0f;
        int node = -1;
    };
    struct QCmp {
        bool operator()(const QItem& a, const QItem& b) const { return a.d2 > b.d2; } // min-heap
    };

    std::priority_queue<QItem, std::vector<QItem>, QCmp> pq;
    pq.push(QItem{dist2_bbox_(nodes_[static_cast<std::size_t>(root_)], q), root_});

    std::vector<std::pair<float, int>> hits; // (d2, idx)
    hits.reserve(256);

    int visited = 0;

    while (!pq.empty()) {
        const QItem cur = pq.top();
        pq.pop();

        if (cur.d2 > r2) continue;
        if (visited >= max_visits) break;
        ++visited;

        const Node& n = nodes_[static_cast<std::size_t>(cur.node)];
        const float d2p = dist2_point_(points_[static_cast<std::size_t>(n.idx)], q);
        if (d2p <= r2) hits.push_back({d2p, n.idx});

        if (n.left >= 0) {
            const float d2b = dist2_bbox_(nodes_[static_cast<std::size_t>(n.left)], q);
            if (d2b <= r2) pq.push(QItem{d2b, n.left});
        }
        if (n.right >= 0) {
            const float d2b = dist2_bbox_(nodes_[static_cast<std::size_t>(n.right)], q);
            if (d2b <= r2) pq.push(QItem{d2b, n.right});
        }
    }

    std::vector<Neighbor> out;
    out.reserve(hits.size());
    std::sort(hits.begin(), hits.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (const auto& [d2, idx] : hits) out.push_back({idx, std::sqrt(d2)});
    return out;
}

} // namespace kdtree2
