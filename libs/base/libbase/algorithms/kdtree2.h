#pragma once

#include <utility>
#include <vector>

#include <libbase/point2.h> // expects libbase::point2f with fields x,y

namespace kdtree2 {

// index in original array + distance (Euclidean) to query point
using Neighbor = std::pair<int, float>;

struct SearchParams {
    // Limits how many KD-tree nodes may be expanded (popped from the queue).
    // Smaller -> faster, more approximate. Larger -> closer to exact.
    int max_nodes_visited = 256;

    // Approximation factor (0 -> exact w.r.t. stopping criterion, but still limited by max_nodes_visited).
    // Typical: 0..0.2. Larger -> more approximate.
    float eps = 0.0f;
};

class KDTree2 {
public:
    KDTree2() = default;
    explicit KDTree2(std::vector<point2f> points);

    void build(std::vector<point2f> points);

    int size() const noexcept;

    // Approximate KNN: returns up to k neighbors sorted by ascending distance.
    std::vector<Neighbor> knn_search(const point2f& q, int k, const SearchParams& params = SearchParams{}) const;

    // Approximate radius: returns neighbors within radius (Euclidean) sorted by ascending distance.
    std::vector<Neighbor> radius_search(const point2f& q, float radius, const SearchParams& params = SearchParams{}) const;

private:
    struct Node {
        int axis = 0;   // 0->x, 1->y
        int idx = -1;   // index into points_
        int left = -1;
        int right = -1;

        // Bounding box of subtree
        float minx = 0, miny = 0, maxx = 0, maxy = 0;
    };

    std::vector<point2f> points_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;
    int root_ = -1;

    int build_rec_(int l, int r);
    void update_bbox_from_children_(int node_id);

    static float dist2_point_(const point2f& a, const point2f& b);
    static float dist2_bbox_(const Node& n, const point2f& q);
};

} // namespace kdtree2
