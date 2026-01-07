#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <libbase/utils.h>
#include <libbase/runtime_assert.h>
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

// KD-tree approximate params.
inline constexpr int kKnnK = 8;
inline constexpr int kKnnMaxNodesVisited = 512;
inline constexpr float kKnnEps = 0.0f;

// Visualization params.
inline constexpr std::uint32_t kVizSeed = 123;
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
    for (const auto& p : pts) {
        out.push_back(point2f{static_cast<float>(p.x) + 0.5f, static_cast<float>(p.y) + 0.5f});
    }
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
        const auto q = point2f{static_cast<float>(queries[static_cast<std::size_t>(i)].x) + 0.5f,
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
        fs::create_directories(dbg_find);
        fs::create_directories(dbg_simpl);
        fs::create_directories(dbg_nn);

        // --- Load + downscale ---
        image8u input = libimages::load_image(input_path.string());
        input = libimages::downscale_area(input, cfg::kDownscaleRatio);

        libimages::debug_io::dump_image((out_dir / cfg::kOut00).string(), input, true, true);

        // --- 1) Find segments (pixel sets) ---
        find_segments::Params sp; // defaults
        find_segments::DebugParams sd;
        sd.out_dir = dbg_find;
        sd.dump_ext = ".png";
        sd.verbose = false;          // low-level without spam
        sd.force_overwrite = true;
        sd.large_component_min_size = 100;

        const auto segments = find_segments::find_segments(input, sp, &sd);

        const auto segments_transformed = map_vector(segments, [](const find_segments::SegmentPixels &s) { return s.pixels; });
        const auto hi_segments = draw::overlay_segments_pixels(input, segments_transformed, cfg::kVizSeed);
        libimages::debug_io::dump_image((out_dir / cfg::kOut01).string(), hi_segments, true, true);

        // Low-level stage 1
        libimages::debug_io::dump_image((dbg_find / "99_high_overlay.png").string(), hi_segments, true, true);

        // --- 2) Simplify segments to [A,B] ---
        const auto simple = simplify_segments::simplify_segments(segments);
        rassert(simple.size() == segments.size(), "Expected simplify size == segments size", simple.size(), segments.size());

        draw::DrawParams dp;
        dp.seed = cfg::kVizSeed;

        const auto simple_transformed = map_vector(simple, [](const simplify_segments::Segment &s) { return std::pair<point2i, point2i>{s.a, s.b}; });
        const auto hi_simple = draw::overlay_simplified_segments(input, simple_transformed, dp);
        libimages::debug_io::dump_image((out_dir / cfg::kOut02).string(), hi_simple, true, true);

        // Low-level stage 2
        libimages::debug_io::dump_image((dbg_simpl / "00_input.png").string(), input, true, true);
        libimages::debug_io::dump_image((dbg_simpl / "01_segments_pixels_overlay.png").string(), hi_segments, true, true);
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

        std::cout << "segments: " << segments.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
