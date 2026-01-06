#include "find_lines.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>

#include "find_segments.h"

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_vertical_line(int w, int h, int x, int thickness) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            img(j, i, 0) = img(j, i, 1) = img(j, i, 2) = 0;

    for (int j = 5; j < h - 5; ++j) {
        for (int t = -thickness; t <= thickness; ++t) {
            const int xx = x + t;
            if (xx >= 0 && xx < w) {
                img(j, xx, 0) = img(j, xx, 1) = img(j, xx, 2) = 255;
            }
        }
    }
    return img;
}

static void run_case(const fs::path& root, const std::string& name, const image8u& img) {
    const fs::path dir = root / name;
    fs::create_directories(dir);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, true, true);

    find_segments::Params sp;
    sp.gaussian_sigma = 0.8f;
    sp.cluster_KD = 1000.0f;
    sp.cluster_KM = 20000.0f;
    sp.min_segment_pixels = 5; // allow smaller segments in synthetic

    const auto segs = find_segments::find_segments(img, sp, /*debug=*/nullptr);
    rassert(!segs.empty(), "Expected segments", name);

    const auto seg_overlay = find_segments::visualize_segments_overlay(img, segs, {});
    libimages::debug_io::dump_image((dir / "01_segments_overlay.png").string(), seg_overlay, true, true);

    find_lines::Params lp;
    lp.min_candidate_pixels = 10;
    lp.max_angle_diff_deg = 10.0f;
    lp.max_dist_px = 3.0f;
    lp.min_total_votes_pixels = 50;
    lp.max_iterations = 1000;

    const auto lines = find_lines::find_lines(segs, lp);
    rassert(!lines.empty(), "Expected at least one line", name);

    const auto lines_overlay = find_lines::visualize_lines_overlay(img, lines, {});
    libimages::debug_io::dump_image((dir / "02_lines_overlay.png").string(), lines_overlay, true, true);

    std::cout << "[test] " << name << ": segs=" << segs.size() << " lines=" << lines.size() << "\n";
}

int main() {
    try {
        const fs::path root = "debug-unit-tests/find-lines";
        fs::create_directories(root);

        run_case(root, "case00_vertical_line", make_vertical_line(220, 160, 110, 1));
        run_case(root, "case01_vertical_line_thick", make_vertical_line(220, 160, 60, 3));

        std::cout << "[test] OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[test] FAIL: " << e.what() << "\n";
        return 2;
    }
}
