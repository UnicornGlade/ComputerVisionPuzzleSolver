#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <libbase/utils.h>
#include <libbase/runtime_assert.h>
#include <libbase/fast_random.h>
#include <libbase/graph.h>
#include <libbase/algorithms/find_cycles.h>
#include <libbase/algorithms/kdtree2.h>

#include <libimages/algorithms/downscale.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>
#include <libimages/draw.h>

#include "algorithms/find_segments.h"
#include "algorithms/simplify_segments.h"

namespace fs = std::filesystem;

namespace cfg {
// Downscale ratio (same meaning as in your pipeline).
inline constexpr float kDownscaleRatio = 8.0f;

// High-level file names.
inline constexpr const char* kOut00 = "00_input_downscaled.png";
inline constexpr const char* kOut01 = "01_segments_pixels_overlay.png";
inline constexpr const char* kOut02 = "02_simplified_segments_overlay.png";
inline constexpr const char* kOut03 = "03_nn_connections_overlay.png";
inline constexpr const char* kOut04 = "04_knn_graph_overlay.png";
inline constexpr const char* kOut05 = "05_cycles_overlay.png";

// KD-tree approximate params.
inline constexpr int kKnnK = 8;
inline constexpr int kKnnMaxNodesVisited = 512;
inline constexpr float kKnnEps = 0.0f;

// Visualization params.
inline constexpr std::uint32_t kVizSeed = 123;

// Graph construction (NEW):
inline constexpr int kGraphK = 2;               // K nearest neighbors
inline constexpr float kGraphMaxDistPx = 20.0f; // keep only edges with dist <= this

} // namespace cfg

using libimages::image8u;

static std::vector<point2i> extract_As(const std::vector<simplify_segments::Segment>& segs) {
    std::vector<point2i> a;
    a.reserve(segs.size());
    for (const auto& s : segs) a.push_back(s.a);
    return a;
}

static std::vector<point2i> extract_Bs(const std::vector<simplify_segments::Segment>& segs) {
    std::vector<point2i> b;
    b.reserve(segs.size());
    for (const auto& s : segs) b.push_back(s.b);
    return b;
}

// Build KD-tree point array from pixel points (use pixel centers).
static std::vector<point2f> to_kd_points(const std::vector<point2i>& pts) {
    std::vector<point2f> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.push_back(point2f{static_cast<float>(p.x) + 0.5f, static_cast<float>(p.y) + 0.5f});
    return out;
}

// For each query point, find nearest neighbor among candidates excluding same index.
// Returns nn_idx (size N), -1 if not found.
static std::vector<int> nn_excluding_self(const kdtree2::KDTree2& tree,
                                         const std::vector<point2i>& queries,
                                         int k,
                                         const kdtree2::SearchParams& sp) {
    const int n = static_cast<int>(queries.size());
    std::vector<int> nn(n, -1);

    if (n <= 1) return nn;
    k = std::max(1, std::min(k, n));

    for (int i = 0; i < n; ++i) {
        const auto q =
            point2f{static_cast<float>(queries[static_cast<std::size_t>(i)].x) + 0.5f,
                    static_cast<float>(queries[static_cast<std::size_t>(i)].y) + 0.5f};

        const auto knn = tree.knn_search(q, k, sp);

        int best = -1;
        for (const auto& [idx, dist] : knn) {
            (void)dist;
            if (idx < 0 || idx >= n) continue;
            if (idx == i) continue;
            best = idx;
            break;
        }
        nn[static_cast<std::size_t>(i)] = best;
    }
    return nn;
}

static float dist_px(point2i a, point2i b) {
    const float dx = static_cast<float>(a.x - b.x);
    const float dy = static_cast<float>(a.y - b.y);
    return std::sqrt(dx * dx + dy * dy);
}

static void put_pixel_rgb_thick(image8u& img, int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                int thickness) {
    const int w = img.width();
    const int h = img.height();

    const int rad = std::max(0, thickness - 1);
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            const int xx = x + dx;
            const int yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            img(yy, xx, 0) = r;
            img(yy, xx, 1) = g;
            img(yy, xx, 2) = b;
        }
    }
}

static void draw_line_bresenham_rgb(image8u& img, point2i a, point2i b, std::uint8_t r, std::uint8_t g,
                                    std::uint8_t bl, int thickness) {
    int x0 = a.x, y0 = a.y;
    const int x1 = b.x, y1 = b.y;

    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        put_pixel_rgb_thick(img, x0, y0, r, g, bl, thickness);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void draw_arrow_rgb(image8u& img, point2i from, point2i to, std::uint8_t r, std::uint8_t g, std::uint8_t bl,
                           int thickness_line, int thickness_head) {
    draw_line_bresenham_rgb(img, from, to, r, g, bl, thickness_line);

    const float dx = static_cast<float>(to.x - from.x);
    const float dy = static_cast<float>(to.y - from.y);
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1.0f) return;

    const float ux = dx / len;
    const float uy = dy / len;

    const float head_len = 6.0f;
    const float ang = 25.0f * 3.14159265358979323846f / 180.0f;
    const float ca = std::cos(ang);
    const float sa = std::sin(ang);

    const float vx = -ux;
    const float vy = -uy;

    const float hx1 = vx * ca - vy * sa;
    const float hy1 = vx * sa + vy * ca;

    const float hx2 = vx * ca + vy * sa;
    const float hy2 = -vx * sa + vy * ca;

    const point2i p1{to.x + static_cast<int>(std::lround(head_len * hx1)),
                     to.y + static_cast<int>(std::lround(head_len * hy1))};
    const point2i p2{to.x + static_cast<int>(std::lround(head_len * hx2)),
                     to.y + static_cast<int>(std::lround(head_len * hy2))};

    draw_line_bresenham_rgb(img, to, p1, r, g, bl, thickness_head);
    draw_line_bresenham_rgb(img, to, p2, r, g, bl, thickness_head);
}

static float edge_weight_or_die(const graph::Graph& gr, int u, int v) {
    for (const auto& e : gr.adj[static_cast<std::size_t>(u)]) {
        if (e.to == v) return e.w;
    }
    rassert(false, "edge_weight_or_die: missing edge", u, v);
    return 0.0f;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage:\n"
                      << "  app <input.(png|jpg|jpeg)> <output_dir> [debug_root]\n";
            return 1;
        }

        const fs::path input_path = argv[1];
        const fs::path out_dir = argv[2];
        fs::create_directories(out_dir);

        const fs::path debug_root = (argc >= 4) ? fs::path(argv[3]) : (out_dir / "debug-low");
        fs::create_directories(debug_root);

        const fs::path dbg_find = debug_root / "01_find_segments";
        const fs::path dbg_simpl = debug_root / "02_simplify_segments";
        const fs::path dbg_nn = debug_root / "03_nearest_neighbors";
        const fs::path dbg_graph = debug_root / "04_graph_cycles";
        fs::create_directories(dbg_find);
        fs::create_directories(dbg_simpl);
        fs::create_directories(dbg_nn);
        fs::create_directories(dbg_graph);

        // --- Load + downscale ---
        image8u input = libimages::load_image(input_path.string());
        input = libimages::downscale_area(input, cfg::kDownscaleRatio);

        libimages::debug_io::dump_image((out_dir / cfg::kOut00).string(), input, true, true);

        // --- 1) Find segments (pixel sets) ---
        find_segments::Params sp; // defaults
        find_segments::DebugParams sd;
        sd.out_dir = dbg_find;
        sd.dump_ext = ".png";
        sd.verbose = false; // low-level without spam
        sd.force_overwrite = true;
        sd.large_component_min_size = 100;

        const auto segments = find_segments::find_segments(input, sp, &sd);

        const auto segments_transformed =
            map_vector(segments, [](const find_segments::SegmentPixels& s) { return s.pixels; });
        const auto hi_segments = draw::overlay_segments_pixels(input, segments_transformed, cfg::kVizSeed);
        libimages::debug_io::dump_image((out_dir / cfg::kOut01).string(), hi_segments, true, true);

        // Low-level stage 1
        libimages::debug_io::dump_image((dbg_find / "99_high_overlay.png").string(), hi_segments, true, true);

        // --- 2) Simplify segments to [A,B] ---
        const auto simple = simplify_segments::simplify_segments(segments);
        rassert(simple.size() == segments.size(), "Expected simplify size == segments size", simple.size(),
                segments.size());

        draw::DrawParams dp;
        dp.seed = cfg::kVizSeed;

        const auto simple_transformed = map_vector(simple, [](const simplify_segments::Segment& s) {
            return std::pair<point2i, point2i>{s.a, s.b};
        });
        const auto hi_simple = draw::overlay_simplified_segments(input, simple_transformed, dp);
        libimages::debug_io::dump_image((out_dir / cfg::kOut02).string(), hi_simple, true, true);

        // Low-level stage 2
        libimages::debug_io::dump_image((dbg_simpl / "00_input.png").string(), input, true, true);
        libimages::debug_io::dump_image((dbg_simpl / "01_segments_pixels_overlay.png").string(), hi_segments, true,
                                        true);
        libimages::debug_io::dump_image((dbg_simpl / "02_simplified_overlay.png").string(), hi_simple, true, true);

        const auto As = extract_As(simple);
        const auto Bs = extract_Bs(simple);
        libimages::debug_io::dump_image((dbg_simpl / "03_A_points.png").string(),
                                        draw::mask_points(input.width(), input.height(), As), true, true);
        libimages::debug_io::dump_image((dbg_simpl / "04_B_points.png").string(),
                                        draw::mask_points(input.width(), input.height(), Bs), true, true);

        // --- 3) For each B: nearest among all other A ---
        kdtree2::SearchParams knn_sp;
        knn_sp.max_nodes_visited = cfg::kKnnMaxNodesVisited;
        knn_sp.eps = cfg::kKnnEps;

        const auto kd_A = to_kd_points(As);
        const auto kd_B = to_kd_points(Bs);

        kdtree2::KDTree2 treeA(kd_A);
        kdtree2::KDTree2 treeB(kd_B);

        const auto nn_B_to_A = nn_excluding_self(treeA, Bs, cfg::kKnnK, knn_sp);

        // --- 4) For each A: nearest among all other B ---
        const auto nn_A_to_B = nn_excluding_self(treeB, As, cfg::kKnnK, knn_sp);

        // High-level final overlay: step2 overlay + red connections for both directions.
        auto hi_nn = hi_simple;
        draw::draw_nn_connections_red_inplace(hi_nn, simple_transformed, nn_B_to_A, nn_A_to_B, dp.nn_line_thickness);
        libimages::debug_io::dump_image((out_dir / cfg::kOut03).string(), hi_nn, true, true);

        // Low-level stage 3: separate forward/reverse overlays + combined.
        auto nn_forward = hi_simple;
        std::vector<int> empty(simple.size(), -1);
        draw::draw_nn_connections_red_inplace(nn_forward, simple_transformed, nn_B_to_A, empty, dp.nn_line_thickness);
        libimages::debug_io::dump_image((dbg_nn / "00_nn_forward_B_to_A.png").string(), nn_forward, true, true);

        auto nn_reverse = hi_simple;
        draw::draw_nn_connections_red_inplace(nn_reverse, simple_transformed, empty, nn_A_to_B, dp.nn_line_thickness);
        libimages::debug_io::dump_image((dbg_nn / "01_nn_reverse_A_to_B.png").string(), nn_reverse, true, true);

        libimages::debug_io::dump_image((dbg_nn / "02_nn_both.png").string(), hi_nn, true, true);

        // --- 5) NEW: KNN graph (B -> K nearest A within max distance), visualize + cycles ---
        const int n = static_cast<int>(simple.size());
        graph::Graph gr(n);

        // Store chosen outgoing edges for visualization: i -> (j, dist)
        std::vector<std::vector<std::pair<int, float>>> out_edges(static_cast<std::size_t>(n));

        const int K = std::max(0, cfg::kGraphK);
        const float max_d = cfg::kGraphMaxDistPx;

        if (n >= 2 && K > 0) {
            for (int i = 0; i < n; ++i) {
                const point2f q{static_cast<float>(Bs[static_cast<std::size_t>(i)].x) + 0.5f,
                                static_cast<float>(Bs[static_cast<std::size_t>(i)].y) + 0.5f};

                const int k_query = std::min(n, K + 8); // extra to safely skip self
                const auto knn = treeA.knn_search(q, k_query, knn_sp);

                for (const auto& [idx, dist] : knn) {
                    if (idx < 0 || idx >= n) continue;
                    if (idx == i) continue;
                    if (dist > max_d) continue;

                    out_edges[static_cast<std::size_t>(i)].push_back({idx, dist});
                    if (static_cast<int>(out_edges[static_cast<std::size_t>(i)].size()) >= K) break;
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            for (const auto& [j, d] : out_edges[static_cast<std::size_t>(i)]) gr.add_edge(i, j, d);
        }

        // 5.1) High-level visualization: simplified segments + directed graph edges
        auto hi_graph = hi_simple;
        for (int i = 0; i < n; ++i) {
            const point2i fromB = Bs[static_cast<std::size_t>(i)];
            for (const auto& [j, d] : out_edges[static_cast<std::size_t>(i)]) {
                (void)d;
                const point2i toA = As[static_cast<std::size_t>(j)];
                draw_arrow_rgb(hi_graph, fromB, toA, /*R=*/255, /*G=*/0, /*B=*/0, /*line=*/1, /*head=*/1);
            }
        }
        libimages::debug_io::dump_image((out_dir / cfg::kOut04).string(), hi_graph, true, true);
        libimages::debug_io::dump_image((dbg_graph / "00_knn_graph_overlay.png").string(), hi_graph, true, true);

        // 5.2) Find biggest cycles (greedy) and visualize them
        const auto cycles = find_cycles::find_cycles_greedy_bruteforce(gr);

        auto hi_cycles = hi_simple;
        FastRandom rng(cfg::kVizSeed ^ 0xA53F9D1Bu);

        std::vector<float> perimeters;
        perimeters.reserve(cycles.size());

        for (std::size_t ci = 0; ci < cycles.size(); ++ci) {
            const auto& cyc = cycles[ci];
            if (cyc.empty()) continue;

            const std::uint32_t rv = rng.nextU32();
            const std::uint8_t cr = static_cast<std::uint8_t>(rv & 0xFFu);
            const std::uint8_t cg = static_cast<std::uint8_t>((rv >> 8) & 0xFFu);
            const std::uint8_t cb = static_cast<std::uint8_t>((rv >> 16) & 0xFFu);

            float sum_edges = 0.0f;
            float sum_segs = 0.0f;

            for (std::size_t k = 0; k < cyc.size(); ++k) {
                const int u = cyc[k];
                const int v = cyc[(k + 1) % cyc.size()];

                const auto& su = simple[static_cast<std::size_t>(u)];
                const auto& sv = simple[static_cast<std::size_t>(v)];

                // Segment length (A_u -> B_u)
                sum_segs += dist_px(su.a, su.b);

                // Graph edge length (B_u -> A_v)
                sum_edges += edge_weight_or_die(gr, u, v);

                // Visualization: cycle-colored segment + cycle-colored connection
                draw_line_bresenham_rgb(hi_cycles, su.a, su.b, cr, cg, cb, /*thick=*/3);
                draw_line_bresenham_rgb(hi_cycles, su.b, sv.a, cr, cg, cb, /*thick=*/2);
            }

            perimeters.push_back(sum_edges + sum_segs);
        }

        libimages::debug_io::dump_image((out_dir / cfg::kOut05).string(), hi_cycles, true, true);
        libimages::debug_io::dump_image((dbg_graph / "01_cycles_overlay.png").string(), hi_cycles, true, true);

        // 5.3) Console stats
        std::cout << "segments: " << segments.size() << "\n";
        std::cout << "cycles: " << cycles.size() << "\n";
        for (std::size_t i = 0; i < perimeters.size(); ++i) {
            std::cout << "cycle[" << i << "] perimeter_px=" << perimeters[i] << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
