#include "find_segments.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <libbase/runtime_assert.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>

namespace fs = std::filesystem;

using libimages::image8u;

static image8u make_vertical_step_edge(int w, int h) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::uint8_t v = (i < w / 2) ? 0 : 255;
            img(j, i, 0) = v;
            img(j, i, 1) = v;
            img(j, i, 2) = v;
        }
    }
    return img;
}

static image8u make_rectangle_border(int w, int h, int x0, int y0, int x1, int y1) {
    image8u img(w, h, 3);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            img(j, i, 0) = img(j, i, 1) = img(j, i, 2) = 0;

    for (int i = x0; i <= x1; ++i) {
        img(y0, i, 0) = img(y0, i, 1) = img(y0, i, 2) = 255;
        img(y1, i, 0) = img(y1, i, 1) = img(y1, i, 2) = 255;
    }
    for (int j = y0; j <= y1; ++j) {
        img(j, x0, 0) = img(j, x0, 1) = img(j, x0, 2) = 255;
        img(j, x1, 0) = img(j, x1, 1) = img(j, x1, 2) = 255;
    }
    return img;
}

static void run_case(const fs::path &root, const std::string &name, const image8u &img) {
    const fs::path dir = root / name;
    fs::create_directories(dir);

    libimages::debug_io::dump_image((dir / "00_input.png").string(), img, /*verbose=*/true, /*force=*/true);

    find_segments::Params p;
    // Keep defaults, but allow smaller min size for tiny synthetic images if needed:
    p.min_segment_pixels = 10;

    const std::vector<find_segments::SegmentPixels> segs = find_segments::find_segments(img, p, /*debug=*/nullptr);

    // Basic sanity
    rassert(!segs.empty(), "Expected at least one segment", name, segs.size());

    find_segments::VisualizeParams vp;
    vp.min_pixels = 2;
    vp.min_median_magnitude = 0.0f;
    vp.max_median_angle_deviation_deg = 180.0f;
    vp.seed = 123;

    const image8u overlay = find_segments::visualize_segments_overlay(img, segs, vp);
    libimages::debug_io::dump_image((dir / "01_segments_overlay.png").string(), overlay, /*verbose=*/true,
                                    /*force=*/true);

    // Also dump only "large-ish" ones for quick viewing.
    find_segments::VisualizeParams vp_large = vp;
    vp_large.min_pixels = 50;
    const image8u overlay_large = find_segments::visualize_segments_overlay(img, segs, vp_large);
    libimages::debug_io::dump_image((dir / "02_segments_overlay_large.png").string(), overlay_large, /*verbose=*/true,
                                    /*force=*/true);

    std::cout << "[test] " << name << ": segments=" << segs.size() << "\n";
}

int main() {
    try {
        const fs::path root = "debug-unit-tests";
        fs::create_directories(root);

        run_case(root, "case00_vertical_step_edge", make_vertical_step_edge(160, 120));
        run_case(root, "case01_rectangle_border", make_rectangle_border(200, 150, 30, 25, 170, 120));

        std::cout << "[test] OK\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[test] FAIL: " << e.what() << "\n";
        return 2;
    }
}
